// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Entry tables (entry_table.hpp): a kernel gets the same entries, in the same
// order, from the table of the mesh and from the traversal of the set
// expression; the mesh records a table once per key and drops it when its cells
// change; and the ghosts update_ghost_mr fills through them hold the same bits
// as with the field operators they replace.

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/domain_builder.hpp>
#include <samurai/entry_table.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_dispatch.hpp>

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
                                                emit(make_rows(mesh,
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
                                                CellGroup group{first, emit.new_range()};
                                                for (std::size_t k = 1; k < i.size(); ++k)
                                                {
                                                    emit.add(group.others, first + k);
                                                }
                                                emit(group);
                                            });
            }
        };

        // Counts the traversals of its set expression: one per table recorded,
        // or one per call without tables.
        struct counted_cells
        {
            using entry_type = Rows<2>;

            std::size_t level;

            inline static std::atomic<std::size_t> traversals{0};

            void produce(const auto& mesh, auto& emit) const
            {
                ++traversals;
                cell_and_next{level}.produce(mesh, emit);
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
                               [&](const auto& rows)
                               {
                                   push(rows.len);
                                   push(rows.at[0]);
                                   push(rows.at[1]);
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
                                   for (const auto cell : cells_of(pool, group.others))
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
                                  auto& mesh          = u.mesh();
                                  const auto level    = mesh.max_level();
                                  const auto expected = seen(mesh, cell_and_next{level});
                                  auto& traversals    = counted_cells::traversals;
                                  traversals          = 0;

                                  EXPECT_EQ(seen(mesh, counted_cells{level}), expected);
                                  EXPECT_EQ(seen(mesh, counted_cells{level}), expected);
                                  EXPECT_EQ(traversals, 1u); // recorded once

                                  // same key type, other member values: another table
                                  seen(mesh, counted_cells{level - 1});
                                  EXPECT_EQ(traversals, 2u);

                                  // without tables, the set expression is traversed at each call
                                  {
                                      const WithEntryTables off(false);
                                      EXPECT_EQ(seen(mesh, counted_cells{level}), expected);
                                      EXPECT_EQ(seen(mesh, counted_cells{level}), expected);
                                      EXPECT_EQ(traversals, 4u);
                                  }
                                  seen(mesh, counted_cells{level});
                                  EXPECT_EQ(traversals, 4u); // the table is still there

                                  // a copy records its own tables
                                  auto copy = mesh;
                                  EXPECT_EQ(seen(copy, counted_cells{level}), expected);
                                  seen(copy, counted_cells{level});
                                  EXPECT_EQ(traversals, 5u);

                                  // the tables follow the cells they were recorded on
                                  std::remove_cvref_t<decltype(mesh)> other;
                                  mesh.swap(other);
                                  EXPECT_EQ(seen(other, counted_cells{level}), expected);
                                  mesh.swap(other);
                                  EXPECT_EQ(seen(mesh, counted_cells{level}), expected);
                                  EXPECT_EQ(traversals, 5u);
                              });
    }

    // Several threads can read the tables of one mesh, recorded on first use.
    TEST(entry_table, tables_read_from_several_threads)
    {
        with_adapted_field<2>(false,
                              [](auto& u)
                              {
                                  const auto& adapted = u.mesh();
                                  std::vector<std::vector<double>> expected;
                                  {
                                      const WithEntryTables off(false);
                                      for (std::size_t level = adapted.min_level(); level <= adapted.max_level(); ++level)
                                      {
                                          expected.push_back(seen(adapted, cell_and_next{level}));
                                          expected.push_back(seen(adapted, weighted_cell_and_next{level}));
                                      }
                                  }

                                  const WithEntryTables on(true);
                                  const auto mesh                 = adapted; // a copy: no table yet
                                  constexpr std::size_t n_threads = 8;
                                  std::array<bool, n_threads> same{};
                                  {
                                      std::vector<std::jthread> threads;
                                      for (std::size_t t = 0; t < n_threads; ++t)
                                      {
                                          threads.emplace_back(
                                              [&, t]
                                              {
                                                  bool ok = true;
                                                  for (std::size_t repeat = 0; repeat < 20; ++repeat)
                                                  {
                                                      std::size_t k = 0;
                                                      for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                                                      {
                                                          ok = ok && seen(mesh, cell_and_next{level}) == expected[k++];
                                                          ok = ok && seen(mesh, weighted_cell_and_next{level}) == expected[k++];
                                                      }
                                                  }
                                                  same[t] = ok;
                                              });
                                      }
                                  }
                                  for (const bool ok : same)
                                  {
                                      EXPECT_TRUE(ok);
                                  }
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
                                           [&](const auto& rows)
                                           {
                                               detail::project_entry(s_new, rows);
                                               detail::project_entry(v_new, rows);
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
                            for_each_entry(mesh,
                                           detail::prediction_ghosts<dim, radius>{level},
                                           [&](const auto& entry)
                                           {
                                               detail::predict_entry<radius>(s_new, entry);
                                               detail::predict_entry<radius>(v_new, entry);
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
        void check_outer_ghost_entries(bool front = false)
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config                 = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(3);
            if (!front)
            {
                config.disable_minimal_ghost_width();
            }
            auto mesh = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a blob that reaches the boundary, or a front that crosses it: the levels change along it
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              if (front)
                              {
                                  u[cell] = std::tanh(100 * (cell.center(0) - 0.5));
                                  return;
                              }
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
                        const auto table = detail::record_entries(adapted, detail::bc_projection_ghosts<dim>{level, dir, 1, 0});
                        n_projected += table.entries.size();
                        n_children += table.pool.size();
                        for (std::size_t layer = 2; layer <= ghost_width; ++layer)
                        {
                            // layers 2 and 3: stencils of 4 and 6 cells
                            if (layer == 2)
                            {
                                n_at_cells += detail::record_entries(adapted, detail::extrapolated_ghosts_at_cells<dim, 4>{level, dir, layer})
                                                  .entries.size();
                                n_at_ghosts += detail::record_entries(adapted,
                                                                      detail::extrapolated_ghosts_at_inner_ghosts<dim, 4>{level, dir, layer})
                                                   .entries.size();
                            }
                            else if (layer == 3)
                            {
                                n_at_cells += detail::record_entries(adapted, detail::extrapolated_ghosts_at_cells<dim, 6>{level, dir, layer})
                                                  .entries.size();
                                n_at_ghosts += detail::record_entries(adapted,
                                                                      detail::extrapolated_ghosts_at_inner_ghosts<dim, 6>{level, dir, layer})
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

    namespace
    {
        // The direct walk of a key gives the entries of its set expression, in
        // the same order, and does not give up. Returns the number of entries.
        template <class Mesh, class Key>
        std::size_t expect_walk_matches_traversal(const Mesh& mesh, const Key& key, const char* what)
        {
            using entry_t = typename Key::entry_type;
            detail::EntryTable<entry_t> walked;
            detail::EntryRecorder<entry_t> recorder(walked);
            EXPECT_TRUE(key.walk(mesh, recorder)) << what << " gave up at level " << key.level;
            const auto traversed = detail::record_entries(mesh, key, false);
            EXPECT_EQ(walked.entries.size(), traversed.entries.size()) << what << " at level " << key.level;
            if (walked.entries.size() == traversed.entries.size() && !walked.entries.empty())
            {
                EXPECT_EQ(std::memcmp(walked.entries.data(), traversed.entries.data(), walked.entries.size() * sizeof(entry_t)), 0)
                    << what << " at level " << key.level;
            }
            EXPECT_EQ(walked.pool, traversed.pool) << what << " at level " << key.level;
            return traversed.entries.size();
        }

        // What the adapted mesh is refined around.
        enum class Refined
        {
            corner_blob, // a blob that reaches a corner of the domain
            front,       // a front normal to x, that crosses the boundaries along x
        };

        template <std::size_t dim>
        void check_walks(int stencil_radius, Refined refined = Refined::corner_blob)
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config                 = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(stencil_radius);
            if (refined == Refined::corner_blob)
            {
                config.disable_minimal_ghost_width();
            }
            auto mesh = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // in both cases, the levels change along the boundary
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              if (refined == Refined::front)
                              {
                                  u[cell] = std::tanh(100 * (cell.center(0) - 0.5));
                                  return;
                              }
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.15) * (cell.center(d) - 0.15);
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - 0.2));
                          });
            make_bc<Neumann<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            const auto& adapted    = u.mesh();
            const auto ghost_width = static_cast<std::size_t>(adapted.ghost_width());
            ASSERT_GE(ghost_width, static_cast<std::size_t>(stencil_radius));

            std::size_t n_projection      = 0;
            std::size_t n_prediction      = 0;
            std::size_t n_at_cells        = 0;
            std::size_t n_corner_copies   = 0;
            std::size_t n_corner_diagonal = 0;
            std::size_t n_corner_others   = 0;
            const std::size_t lowest      = adapted.min_level() - 1;
            for (std::size_t level = lowest; level <= adapted.max_level(); ++level)
            {
                for_each_cartesian_direction<dim>(
                    [&](const auto& direction)
                    {
                        const auto dir = direction_code<dim>(direction);
                        for (std::size_t layer = 1; layer <= ghost_width; ++layer)
                        {
                            if (level < adapted.max_level() && layer <= (ghost_width + 1) / 2)
                            {
                                n_projection += expect_walk_matches_traversal(adapted,
                                                                              detail::bc_projection_ghosts<dim>{level, dir, layer, 0},
                                                                              "project_bc");
                            }
                            if (level > adapted.min_level())
                            {
                                n_prediction += expect_walk_matches_traversal(adapted,
                                                                              detail::bc_prediction_ghosts<dim>{level, dir, layer, 0},
                                                                              "predict_bc");
                            }
                            if (level < adapted.min_level())
                            {
                                continue;
                            }
                            detail::for_each_extrapolated_ghost_layer(
                                static_cast<int>(layer),
                                static_cast<int>(layer),
                                [&](auto stencil_size_, std::size_t)
                                {
                                    static constexpr std::size_t size = decltype(stencil_size_)::value;
                                    n_at_cells += expect_walk_matches_traversal(
                                        adapted,
                                        detail::extrapolated_ghosts_at_cells<dim, size>{level, dir, layer},
                                        "extrapolation at cells");
                                    if (layer > 1)
                                    {
                                        expect_walk_matches_traversal(
                                            adapted,
                                            detail::extrapolated_ghosts_at_inner_ghosts<dim, size>{level, dir, layer},
                                            "extrapolation at inner ghosts");
                                    }
                                });
                        }
                    });
                if constexpr (dim > 1)
                {
                    if (level < adapted.min_level())
                    {
                        continue;
                    }
                    for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            const auto dir = direction_code<dim>(direction);
                            n_corner_copies += expect_walk_matches_traversal(adapted,
                                                                             detail::corner_ghosts_below<dim>{level, dir},
                                                                             "corner copies");
                            n_corner_others += expect_walk_matches_traversal(adapted,
                                                                             detail::corner_off_diagonal_ghosts<dim>{level, dir},
                                                                             "off-diagonal corner ghosts");
                            for (std::size_t layer = 1; layer <= ghost_width; ++layer)
                            {
                                if (2 * layer <= detail::max_stencil_size_implemented_PE)
                                {
                                    dispatch_static<1, detail::max_stencil_size_implemented_PE / 2>(
                                        layer,
                                        [&](auto layer_)
                                        {
                                            static constexpr std::size_t size = 2 * decltype(layer_)::value;
                                            n_corner_diagonal += expect_walk_matches_traversal(
                                                adapted,
                                                detail::corner_diagonal_ghosts<dim, size>{level, dir, layer},
                                                "diagonal corner ghosts");
                                        });
                                }
                                else
                                {
                                    n_corner_diagonal += expect_walk_matches_traversal(
                                        adapted,
                                        detail::corner_diagonal_ghosts<dim, 2>{level, dir, layer},
                                        "mirrored diagonal corner ghosts");
                                }
                            }
                        });
                }
            }
            // the comparisons above had entries to compare
            EXPECT_GT(n_projection, 0u);
            EXPECT_GT(n_at_cells, 0u);
            // predict_bc has ghosts to fill where a front crosses the boundary: next to a coarser boundary cell
            if (dim > 1 && refined == Refined::front)
            {
                EXPECT_GT(n_prediction, 0u);
            }
            if constexpr (dim > 1)
            {
                EXPECT_GT(n_corner_copies, 0u);
                EXPECT_GT(n_corner_diagonal, 0u);
                // with one ghost layer, the corner block is its diagonal ghost
                EXPECT_EQ(n_corner_others > 0, ghost_width > 1);
            }
        }
    }

    TEST(entry_table, walks_same_entries_as_set_traversal_1d)
    {
        check_walks<1>(3);
    }

    TEST(entry_table, walks_same_entries_as_set_traversal_2d)
    {
        check_walks<2>(1);
        check_walks<2>(3);
        check_walks<2>(4); // the diagonal corner ghost of the layer 4 is mirrored
        check_walks<2>(2, Refined::front);
        check_walks<2>(3, Refined::front);
    }

    TEST(entry_table, walks_same_entries_as_set_traversal_3d)
    {
        check_walks<3>(2);
        check_walks<3>(2, Refined::front);
    }

    // The keys where there is nothing to emit, and the walks that give up.
    TEST(entry_table, empty_and_refused_walks)
    {
        static constexpr std::size_t dim = 2;
        auto config                      = mesh_config<dim>().min_level(3).max_level(3).max_stencil_radius(2);
        const auto mesh                  = mra::make_mesh(Box<double, dim>({0., 0.}, {1., 1.}), config);
        using mesh_t                     = std::remove_cvref_t<decltype(mesh)>;
        using mesh_id_t                  = typename mesh_t::mesh_id_t;
        using lca_t                      = typename mesh_t::lca_type;
        const DirectionVector<dim> diagonal{1, 1};
        const DirectionVector<dim> cartesian{1, 0};

        // no corner ghost below the level 0
        const detail::corner_ghosts_below<dim> below_level_0{0, direction_code<dim>(diagonal)};
        EXPECT_TRUE(detail::record_entries(mesh, below_level_0, false).entries.empty());
        EXPECT_TRUE(detail::record_entries(mesh, below_level_0).entries.empty());

        // a cartesian direction has no off-diagonal ghost
        std::size_t n_offsets = 0;
        detail::for_each_off_diagonal_offset<dim>(cartesian,
                                                  2,
                                                  [&](int, const auto&)
                                                  {
                                                      ++n_offsets;
                                                  });
        EXPECT_EQ(n_offsets, 0u);
        detail::for_each_off_diagonal_offset<dim>(diagonal,
                                                  2,
                                                  [&](int, const auto&)
                                                  {
                                                      ++n_offsets;
                                                  });
        EXPECT_EQ(n_offsets, 2u); // one per layer

        // blocks: an empty array is not a block, and a block does not hold what sticks out of it
        EXPECT_FALSE(detail::as_block(lca_t(2)).has_value());
        const auto& cells = mesh[mesh_id_t::cells][3]; // the uniform mesh
        const auto block  = detail::as_block(cells);
        ASSERT_TRUE(block.has_value());
        EXPECT_TRUE(detail::block_holds(*block, cells));
        EXPECT_FALSE(detail::block_holds(detail::translate(*block, cartesian), cells));
        EXPECT_FALSE(detail::cell_index(cells, {block->hi[0], block->lo[1]}) >= 0);
        EXPECT_TRUE(detail::cell_index(cells, {block->lo[0], block->lo[1]}) >= 0);

        // the corner is only known up to its own level
        EXPECT_TRUE(detail::corner_block(mesh, diagonal, mesh.corner(diagonal).level()).has_value());
        EXPECT_FALSE(detail::corner_block(mesh, diagonal, mesh.corner(diagonal).level() + 1).has_value());

        // so the walks of the corner keys give up there
        const auto too_fine = mesh.corner(diagonal).level() + 1;
        const auto dir      = direction_code<dim>(diagonal);
        detail::EntryTable<Cells<2>> walked;
        detail::EntryRecorder<Cells<2>> recorder(walked);
        EXPECT_FALSE((detail::corner_ghosts_below<dim>{too_fine, dir}).walk(mesh, recorder));
        EXPECT_FALSE((detail::corner_diagonal_ghosts<dim, 2>{too_fine, dir, 1}).walk(mesh, recorder));
        EXPECT_FALSE((detail::corner_off_diagonal_ghosts<dim>{too_fine, dir}).walk(mesh, recorder));
        EXPECT_TRUE(walked.entries.empty());
    }

    // The NaN checks of project_bc and predict_bc name a cell by its level and
    // its coordinates, found back from its storage index.
    TEST(entry_table, cell_described_from_its_storage_index)
    {
        with_adapted_field<2>(
            false,
            [](auto& u)
            {
                const auto& mesh    = u.mesh();
                std::size_t n_cells = 0;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  if (n_cells++ % 97 != 0)
                                  {
                                      return;
                                  }
                                  std::ostringstream expected;
                                  expected << "storage index " << cell.index << " (level " << cell.level << ", i = " << cell.indices[0]
                                           << ", index = ";
                                  const auto described = detail::describe_cell(mesh, static_cast<std::size_t>(cell.index));
                                  EXPECT_EQ(described.substr(0, expected.str().size()), expected.str());
                                  EXPECT_NE(described.find(std::to_string(cell.indices[1])), std::string::npos);
                              });
                EXPECT_GT(n_cells, 97u);
                // an index that is no cell of the mesh
                EXPECT_EQ(detail::describe_cell(mesh, u.array().size() + 1), "storage index " + std::to_string(u.array().size() + 1));
            });
    }

    // A store keeps its tables when moved or swapped, and never copies them.
    TEST(entry_table, store_copy_move_swap)
    {
        with_adapted_field<2>(false,
                              [](auto& u)
                              {
                                  const auto& mesh = u.mesh();
                                  const counted_cells key{mesh.max_level()};
                                  auto& traversals = counted_cells::traversals;
                                  traversals       = 0;

                                  detail::EntryTableStore store;
                                  const auto* table = &store.get(key, mesh);
                                  EXPECT_EQ(&store.get(key, mesh), table);
                                  EXPECT_EQ(traversals, 1u);

                                  detail::EntryTableStore copy(store);
                                  copy.get(key, mesh);
                                  EXPECT_EQ(traversals, 2u); // the copy had no table
                                  copy = store;
                                  copy.get(key, mesh);
                                  EXPECT_EQ(traversals, 3u); // it dropped its own, and took none
                                  auto& same = store;
                                  store      = same;
                                  EXPECT_EQ(&store.get(key, mesh), table);

                                  detail::EntryTableStore moved(std::move(store));
                                  EXPECT_EQ(&moved.get(key, mesh), table);
                                  detail::EntryTableStore assigned;
                                  assigned.get(key, mesh);
                                  EXPECT_EQ(traversals, 4u);
                                  assigned = std::move(moved);
                                  EXPECT_EQ(&assigned.get(key, mesh), table);
                                  assigned.swap(assigned);
                                  auto& itself = assigned;
                                  assigned     = std::move(itself);
                                  EXPECT_EQ(&assigned.get(key, mesh), table);
                                  EXPECT_EQ(traversals, 4u);

                                  assigned.clear();
                                  assigned.get(key, mesh);
                                  EXPECT_EQ(traversals, 5u);
                              });
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_1d)
    {
        check_outer_ghost_entries<1>();
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_2d)
    {
        check_outer_ghost_entries<2>();
        check_outer_ghost_entries<2>(true);
    }

    TEST(entry_table, outer_ghosts_same_with_and_without_tables_3d)
    {
        check_outer_ghost_entries<3>();
        check_outer_ghost_entries<3>(true);
    }

    // On a domain with a hole, no walk applies and the ghosts the B.C. fills in
    // other directions are left out of project_bc and predict_bc: the tables,
    // recorded from the set expressions, give the same bits as no table.
    TEST(entry_table, outer_ghosts_same_with_and_without_tables_hole)
    {
#ifdef SAMURAI_WITH_MPI
        GTEST_SKIP() << "DomainBuilder is not implemented with MPI";
#endif
        static constexpr std::size_t dim = 2;
        DomainBuilder<dim> domain({0., 0.}, {1., 1.});
        domain.remove({0.25, 0.25}, {0.75, 0.75});
        auto config = mesh_config<dim>().min_level(4).max_level(6).max_stencil_radius(1).scaling_factor(1.).disable_args_parse();
        auto mesh   = mra::make_mesh(domain, config);

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = std::tanh(100 * (cell.center(0) - 0.85));
                      });
        make_bc<Neumann<1>>(u, 0.);
        make_MRAdapt(u)(mra_config().epsilon(1e-3));

        const auto& adapted = u.mesh();
        ASSERT_FALSE(adapted.domain().is_box());
        EXPECT_FALSE(detail::box_domain_block(adapted, adapted.max_level(), false).has_value());

        auto s = make_scalar_field<double>("s", u.mesh());
        auto v = make_vector_field<double, 3>("v", u.mesh());
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

        // the keys carry the B.C. ghosts left out, and their walks give up
        std::size_t n_projection = 0;
        for (std::size_t level = adapted.min_level() - 1; level < adapted.max_level(); ++level)
        {
            for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    const detail::bc_projection_ghosts<dim> key{level, direction_code<dim>(direction), 1, 1};
                    detail::EntryTable<detail::BcProjectionEntry> walked;
                    detail::EntryRecorder<detail::BcProjectionEntry> recorder(walked);
                    EXPECT_FALSE(key.walk(adapted, recorder));
                    n_projection += detail::record_entries(adapted, key).entries.size();
                });
        }
        EXPECT_GT(n_projection, 0u);
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
