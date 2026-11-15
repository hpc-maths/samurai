// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Entry tables (entry_table.hpp): a kernel gets the same entries, in the same
// order, from the table of the mesh and from the traversal of the set
// expression; the mesh records a table once per key and drops it when its cells
// change; and the ghosts update_ghost_mr fills through them hold the same bits
// as with the field operators they replace.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/entry_table.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    namespace
    {
        // Sets entry_tables_enabled() for the time of a scope.
        class WithEntryTables
        {
          public:

            explicit WithEntryTables(bool enabled)
                : m_previous(entry_tables_enabled())
            {
                entry_tables_enabled() = enabled;
            }

            ~WithEntryTables()
            {
                entry_tables_enabled() = m_previous;
            }

            WithEntryTables(const WithEntryTables&)            = delete;
            WithEntryTables& operator=(const WithEntryTables&) = delete;

          private:

            bool m_previous;
        };

        // Every stored value, ghosts included, set to a value that differs from
        // one storage slot to the next.
        template <class Field>
        void fill_all(Field& f, double seed)
        {
            auto* data       = f.data();
            const auto count = f.array().size();
            for (std::size_t k = 0; k < count; ++k)
            {
                data[k] = std::sin(seed + 0.37 * static_cast<double>(k));
            }
        }

        template <class Field>
        bool same_bits(const Field& a, const Field& b)
        {
            return std::memcmp(a.data(), b.data(), a.array().size() * sizeof(*a.data())) == 0;
        }

        // A field adapted around a blob, so that every level has ghosts.
        template <std::size_t dim, class Func>
        void with_adapted_field(bool periodic, Func&& f)
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).periodic(periodic).disable_minimal_ghost_width();
            auto mesh = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.4) * (cell.center(d) - 0.4);
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - 0.2));
                          });
            if (!periodic)
            {
                make_bc<Dirichlet<1>>(u, 0.);
            }
            make_MRAdapt(u)(mra_config().epsilon(1e-3));
            f(u);
        }

        //////////////////////////////////////////////////////////////////
        // Keys a user could write
        //////////////////////////////////////////////////////////////////

        // The cells of `level` whose next cell along x is also a cell.
        template <class Mesh, class Func>
        void for_each_cell_row_with_next(const Mesh& mesh, std::size_t level, Func&& f)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            xt::xtensor_fixed<int, xt::xshape<Mesh::dim>> back;
            back.fill(0);
            back[0]  = -1;
            auto set = intersection(mesh[mesh_id_t::cells][level], translate(mesh[mesh_id_t::cells][level], back));
            set(f);
        }

        struct cell_and_next
        {
            using entry_type = Rows<2>;

            std::size_t level;

            void produce(const auto& mesh, auto& emit) const
            {
                for_each_cell_row_with_next(mesh,
                                            level,
                                            [&](const auto& i, const auto& index)
                                            {
                                                emit(rows(mesh,
                                                          i.size(),
                                                          {
                                                              {level, i.start,     index},
                                                              {level, i.start + 1, index}
                                                }));
                                            });
            }
        };

        // Same set expression, other entries: a struct of its own, one per cell.
        struct WeightedPair
        {
            std::size_t cell;
            std::size_t next;
            double weight;
        };

        struct weighted_cell_and_next
        {
            using entry_type = WeightedPair;

            std::size_t level;

            void produce(const auto& mesh, auto& emit) const
            {
                for_each_cell_row_with_next(mesh,
                                            level,
                                            [&](const auto& i, const auto& index)
                                            {
                                                const auto first = memory_offset(mesh, {level, i.start, index});
                                                for (std::size_t k = 0; k < i.size(); ++k)
                                                {
                                                    emit(WeightedPair{first + k, first + k + 1, 1. / static_cast<double>(k + 1)});
                                                }
                                            });
            }
        };

        // Same set expression again, with a variable number of cells per entry.
        struct CellGroup
        {
            std::size_t first_cell;
            PoolRange others;
        };

        struct grouped_cells
        {
            using entry_type = CellGroup;

            std::size_t level;

            void produce(const auto& mesh, auto& emit) const
            {
                for_each_cell_row_with_next(mesh,
                                            level,
                                            [&](const auto& i, const auto& index)
                                            {
                                                const auto first = memory_offset(mesh, {level, i.start, index});
                                                CellGroup group{
                                                    first,
                                                    {emit.pool().size(), 0}
                                                };
                                                for (std::size_t k = 1; k < i.size(); ++k)
                                                {
                                                    emit.pool().push_back(first + k);
                                                    ++group.others.count;
                                                }
                                                emit(group);
                                            });
            }
        };

        // A key that refuses to be kept.
        struct never_kept
        {
            using entry_type = Rows<2>;

            std::size_t level;

            void produce(const auto& mesh, auto& emit) const
            {
                cell_and_next{level}.produce(mesh, emit);
            }

            bool keep(const auto&) const
            {
                return false;
            }
        };

        // What a kernel sees, flattened.
        template <class Mesh, class Key>
        std::vector<double> seen(const Mesh& mesh, const Key& key)
        {
            std::vector<double> values;
            auto push = [&](auto v)
            {
                values.push_back(static_cast<double>(v));
            };
            using entry_t = typename Key::entry_type;
            if constexpr (std::same_as<entry_t, Rows<2>>)
            {
                for_each_entry(mesh,
                               key,
                               [&](const auto& run)
                               {
                                   push(run.len);
                                   push(run.at[0]);
                                   push(run.at[1]);
                               });
            }
            else if constexpr (std::same_as<entry_t, WeightedPair>)
            {
                for_each_entry(mesh,
                               key,
                               [&](const WeightedPair& pair)
                               {
                                   push(pair.cell);
                                   push(pair.next);
                                   push(pair.weight);
                               });
            }
            else
            {
                for_each_entry(mesh,
                               key,
                               [&](const CellGroup& group, std::span<const std::size_t> pool)
                               {
                                   push(group.first_cell);
                                   for (const auto cell : pool.subspan(group.others.first, group.others.count))
                                   {
                                       push(cell);
                                   }
                               });
            }
            return values;
        }

        // Returns how many values the kernel saw.
        template <class Mesh, class Key>
        std::size_t expect_same_entries(const Mesh& mesh, const Key& key)
        {
            std::vector<double> from_sets;
            {
                const WithEntryTables off(false);
                from_sets = seen(mesh, key);
            }
            const WithEntryTables on(true);
            EXPECT_EQ(seen(mesh, key), from_sets); // records the table
            EXPECT_EQ(seen(mesh, key), from_sets); // reads it
            return from_sets.size();
        }
    }

    TEST(entry_table, user_keys_same_entries_as_set_traversal)
    {
        with_adapted_field<2>(false,
                              [](auto& u)
                              {
                                  const auto& mesh       = u.mesh();
                                  std::size_t n_rows     = 0;
                                  std::size_t n_weighted = 0;
                                  std::size_t n_grouped  = 0;
                                  for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                                  {
                                      n_rows += expect_same_entries(mesh, cell_and_next{level});
                                      n_weighted += expect_same_entries(mesh, weighted_cell_and_next{level});
                                      n_grouped += expect_same_entries(mesh, grouped_cells{level});
                                  }
                                  // the comparisons above had entries to compare
                                  EXPECT_GT(n_rows, 0u);
                                  EXPECT_GT(n_weighted, 0u);
                                  EXPECT_GT(n_grouped, n_rows / 3); // some groups have cells in the pool
                              });
    }

    TEST(entry_table, one_table_per_key)
    {
        with_adapted_field<2>(false,
                              [](auto& u)
                              {
                                  const WithEntryTables on(true);
                                  auto& mesh             = u.mesh();
                                  const auto level       = mesh.max_level();
                                  const auto& store      = mesh.entry_tables();
                                  const std::size_t base = store.size(); // the tables of the ghost updates of the adaptation

                                  seen(mesh, cell_and_next{level});
                                  EXPECT_EQ(store.size(), base + 1);
                                  seen(mesh, cell_and_next{level});
                                  EXPECT_EQ(store.size(), base + 1); // recorded once

                                  // same set expression, other entries: a table of its own
                                  seen(mesh, weighted_cell_and_next{level});
                                  EXPECT_EQ(store.size(), base + 2);

                                  // same key type, other member values: another table
                                  seen(mesh, cell_and_next{level - 1});
                                  EXPECT_EQ(store.size(), base + 3);

                                  // a key that refuses, and disabled tables: nothing kept
                                  EXPECT_FALSE(has_entry_table(mesh, never_kept{level}));
                                  EXPECT_EQ(seen(mesh, never_kept{level}), seen(mesh, cell_and_next{level}));
                                  EXPECT_EQ(store.size(), base + 3);
                                  {
                                      const WithEntryTables off(false);
                                      EXPECT_FALSE(has_entry_table(mesh, grouped_cells{level}));
                                      seen(mesh, grouped_cells{level});
                                      EXPECT_EQ(store.size(), base + 3);
                                  }

                                  // a copy records its own tables
                                  auto copy = mesh;
                                  EXPECT_EQ(copy.entry_tables().size(), 0u);
                                  EXPECT_EQ(seen(copy, cell_and_next{level}), seen(mesh, cell_and_next{level}));
                                  EXPECT_EQ(copy.entry_tables().size(), 1u);

                                  // the tables follow the cells they were recorded on
                                  std::remove_cvref_t<decltype(mesh)> other;
                                  mesh.swap(other);
                                  EXPECT_EQ(mesh.entry_tables().size(), 0u);
                                  EXPECT_EQ(other.entry_tables().size(), base + 3);
                                  mesh.swap(other);
                                  EXPECT_EQ(mesh.entry_tables().size(), base + 3);
                              });
    }

    namespace
    {
        // The ghosts filled through entries against the field operators they
        // replace, bit for bit, at every level, for a scalar and a vector field.
        template <std::size_t dim>
        void check_ghost_entries(bool periodic)
        {
            with_adapted_field<dim>(
                periodic,
                [&](auto& u)
                {
                    using mesh_t                 = std::remove_cvref_t<decltype(u.mesh())>;
                    using mesh_id_t              = typename mesh_t::mesh_id_t;
                    constexpr std::size_t radius = mesh_t::config_t::prediction_stencil_radius;
                    const auto& mesh             = u.mesh();

                    auto s = make_scalar_field<double>("s", u.mesh());
                    auto v = make_vector_field<double, 3>("v", u.mesh());
                    fill_all(s, 0.1);
                    fill_all(v, 0.7);

                    std::size_t n_projected = 0;
                    std::size_t n_predicted = 0;
                    std::size_t n_periodic  = 0;
                    std::vector<std::vector<double>> whole_s;
                    std::vector<std::vector<double>> whole_v;
                    for (const bool tables : {true, false})
                    {
                        const WithEntryTables with(tables);
                        for (std::size_t level = 1; level <= mesh.max_level(); ++level)
                        {
                            auto s_ref = s;
                            auto v_ref = v;
                            auto s_new = s;
                            auto v_new = v;
                            intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1])
                                .on(level - 1)
                                .apply_op(variadic_projection(s_ref, v_ref));
                            for_each_entry(mesh,
                                           detail::projection_ghosts<dim>{level - 1},
                                           [&](const auto& run)
                                           {
                                               detail::project_run(s_new, run);
                                               detail::project_run(v_new, run);
                                               ++n_projected;
                                           });
                            EXPECT_TRUE(same_bits(s_ref, s_new) && same_bits(v_ref, v_new))
                                << "field projection to level " << level - 1 << ", tables " << tables;

                            s_ref            = s;
                            v_ref            = v;
                            s_new            = s;
                            v_new            = v;
                            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
                            intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1])
                                .on(level)
                                .apply_op(variadic_prediction<radius, false>(s_ref, v_ref));
                            detail::PredictionKernel<dim, radius> predict;
                            for_each_entry(mesh,
                                           detail::prediction_ghosts<dim, radius>{level},
                                           [&](const auto& entry)
                                           {
                                               predict(s_new, entry);
                                               predict(v_new, entry);
                                               ++n_predicted;
                                           });
                            EXPECT_TRUE(same_bits(s_ref, s_new) && same_bits(v_ref, v_new))
                                << "prediction at level " << level << ", tables " << tables;

                            s_ref     = s;
                            v_ref     = v;
                            s_new     = s;
                            v_new     = v;
                            auto copy = [&](auto& field)
                            {
                                iterate_over_periodic_ghosts(
                                    level,
                                    field,
                                    [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                                    {
                                        field(level, i_ghosts, index_ghosts) = field(level, i_cells, index_cells);
                                        ++n_periodic;
                                    });
                            };
                            copy(s_ref);
                            copy(v_ref);
                            update_ghost_periodic(level, s_new, v_new);
                            EXPECT_TRUE(same_bits(s_ref, s_new) && same_bits(v_ref, v_new))
                                << "periodic copies at level " << level << ", tables " << tables;
                        }

                        // the whole ghost update, kept to compare both ways
                        auto s_all = s;
                        auto v_all = v;
                        update_ghost_mr(s_all, v_all);
                        whole_s.emplace_back(s_all.data(), s_all.data() + s_all.array().size());
                        whole_v.emplace_back(v_all.data(), v_all.data() + v_all.array().size());
                    }
                    EXPECT_EQ(whole_s[0], whole_s[1]) << "update_ghost_mr with and without tables";
                    EXPECT_EQ(whole_v[0], whole_v[1]) << "update_ghost_mr with and without tables";
                    // the comparisons above had ghosts to compare
                    EXPECT_GT(n_projected, 0u);
                    EXPECT_GT(n_predicted, 0u);
                    EXPECT_EQ(n_periodic > 0, periodic);
                });
        }
    }

    namespace
    {
        // The ghosts outside the domain filled through entries (project_bc and
        // the polynomial extrapolation of the further layers): the same bits
        // with and without tables, and tables that are not empty.
        template <std::size_t dim>
        void check_outer_ghost_entries()
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(3).disable_minimal_ghost_width();
            auto mesh   = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a blob that reaches the boundary, so that its levels change along it
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.15) * (cell.center(d) - 0.15);
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - 0.2));
                          });
            make_bc<Dirichlet<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            const auto& adapted = u.mesh();
            auto s              = make_scalar_field<double>("s", u.mesh());
            auto v              = make_vector_field<double, 3>("v", u.mesh());
            make_bc<Dirichlet<1>>(s, 0.);
            make_bc<Dirichlet<1>>(v, 0., 0., 0.);
            fill_all(s, 0.1);
            fill_all(v, 0.7);

            std::vector<std::vector<double>> whole_s;
            std::vector<std::vector<double>> whole_v;
            for (const bool tables : {true, false})
            {
                const WithEntryTables with(tables);
                auto s_all = s;
                auto v_all = v;
                update_outer_ghosts(s_all, v_all);
                whole_s.emplace_back(s_all.data(), s_all.data() + s_all.array().size());
                whole_v.emplace_back(v_all.data(), v_all.data() + v_all.array().size());
            }
            EXPECT_EQ(whole_s[0], whole_s[1]);
            EXPECT_EQ(whole_v[0], whole_v[1]);

            // what the comparison went through
            std::size_t n_projected = 0;
            std::size_t n_children  = 0;
            std::size_t n_at_cells  = 0;
            std::size_t n_at_ghosts = 0;
            const auto ghost_width  = static_cast<std::size_t>(adapted.ghost_width());
            for (std::size_t level = adapted.min_level(); level <= adapted.max_level(); ++level)
            {
                for_each_cartesian_direction<dim>(
                    [&](const auto& direction)
                    {
                        const auto dir   = direction_code<dim>(direction);
                        const auto table = record_entries(adapted, detail::bc_projection_ghosts<dim>{level, dir, 1, 0});
                        n_projected += table.entries.size();
                        n_children += table.pool.size();
                        for (std::size_t layer = 2; layer <= ghost_width; ++layer)
                        {
                            // layers 2 and 3: stencils of 4 and 6 cells
                            if (layer == 2)
                            {
                                n_at_cells += record_entries(adapted, detail::extrapolated_ghosts_at_cells<dim, 4>{level, dir, layer})
                                                  .entries.size();
                                n_at_ghosts += record_entries(adapted, detail::extrapolated_ghosts_at_inner_ghosts<dim, 4>{level, dir, layer})
                                                   .entries.size();
                            }
                            else if (layer == 3)
                            {
                                n_at_cells += record_entries(adapted, detail::extrapolated_ghosts_at_cells<dim, 6>{level, dir, layer})
                                                  .entries.size();
                                n_at_ghosts += record_entries(adapted, detail::extrapolated_ghosts_at_inner_ghosts<dim, 6>{level, dir, layer})
                                                   .entries.size();
                            }
                        }
                    });
            }
            EXPECT_GE(ghost_width, 3u);
            EXPECT_GT(n_projected, 0u);
            EXPECT_GT(n_children, n_projected); // several children per ghost
            EXPECT_GT(n_at_cells, 0u);
            // n_at_ghosts is 0 on these meshes: no inner ghost of the boundary has its whole stencil in the reference mesh
            static_cast<void>(n_at_ghosts);
        }
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_1d)
    {
        check_outer_ghost_entries<1>();
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_2d)
    {
        check_outer_ghost_entries<2>();
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_3d)
    {
        check_outer_ghost_entries<3>();
    }

    TEST(entry_table, ghosts_same_as_field_operators_1d)
    {
        check_ghost_entries<1>(false);
        check_ghost_entries<1>(true);
    }

    TEST(entry_table, ghosts_same_as_field_operators_2d)
    {
        check_ghost_entries<2>(false);
        check_ghost_entries<2>(true);
    }

    TEST(entry_table, ghosts_same_as_field_operators_3d)
    {
        check_ghost_entries<3>(false);
        check_ghost_entries<3>(true);
    }
}
