// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The tables of update_ghost_mr against the set expressions they replace: on an
// adapted mesh, the field projection, the prediction, the periodic copies and the
// ghosts outside the domain filled from other ghosts (project_bc, predict_bc and
// the corner copies) or by polynomial extrapolation must give the same values, bit for bit, at every level, for a
// scalar and a vector field.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    namespace
    {
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

        template <std::size_t dim>
        void check_tables()
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).disable_minimal_ghost_width();
            auto mesh   = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a blob that refines part of the domain, so that every level has ghosts
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
            make_bc<Dirichlet<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            // the comparison below must have ghosts to compare
            const auto& tables = u.mesh().ghost_tables();
            ASSERT_FALSE(tables.empty());
            std::size_t n_projected = 0;
            std::size_t n_predicted = 0;
            for (const auto& t : tables.projection)
            {
                n_projected += t.dest.size();
            }
            for (const auto& t : tables.prediction)
            {
                n_predicted += t.dest.size();
            }
            ASSERT_GT(n_projected, 0u);
            {
                const auto& mm = u.mesh();
                for (std::size_t l = mm.min_level(); l <= mm.max_level(); ++l)
                {
                    std::cerr << "level " << l << " cells " << mm[std::decay_t<decltype(mm)>::mesh_id_t::cells][l].nb_cells() << " ref "
                              << mm[std::decay_t<decltype(mm)>::mesh_id_t::reference][l].nb_cells();
                    for_each_cartesian_direction<dim>(
                        [&](auto, const auto& dir)
                        {
                            std::size_t n = 0;
                            if (l > mm.min_level())
                            {
                                auto g = intersection(domain_boundary_outer_layer(mm, l - 1, dir, 1),
                                                      mm[std::decay_t<decltype(mm)>::mesh_id_t::reference][l])
                                             .on(l);
                                g(
                                    [&](const auto& i, const auto&)
                                    {
                                        n += i.size();
                                    });
                            }
                            std::cerr << " pred " << n;
                        });
                    std::cerr << "\n";
                }
            }
            ASSERT_GT(n_predicted, 0u);

            auto s = make_scalar_field<double>("s", u.mesh());
            auto v = make_vector_field<double, 3>("v", u.mesh());
            fill_all(s, 0.1);
            fill_all(v, 0.7);

            for (std::size_t level = 1; level <= u.mesh().max_level(); ++level)
            {
                auto s_sets  = s;
                auto v_sets  = v;
                auto s_table = s;
                auto v_table = v;
                detail::project_ghosts_with_sets(level, s_sets, v_sets);
                detail::project_ghosts(level, s_table, v_table);
                EXPECT_TRUE(same_bits(s_sets, s_table) && same_bits(v_sets, v_table)) << "field projection to level " << level - 1;

                s_sets  = s;
                v_sets  = v;
                s_table = s;
                v_table = v;
                detail::predict_ghosts_with_sets(level, s_sets, v_sets);
                detail::predict_ghosts(level, s_table, v_table);
                EXPECT_TRUE(same_bits(s_sets, s_table) && same_bits(v_sets, v_table)) << "prediction at level " << level;
            }
        }

        template <class Field>
        void copy_periodic_with_sets(std::size_t level, Field& f)
        {
            iterate_over_periodic_ghosts(level,
                                         f,
                                         [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                                         {
                                             f(level, i_ghosts, index_ghosts) = f(level, i_cells, index_cells);
                                         });
        }

        // Periodic in every direction, or in x only.
        template <std::size_t dim>
        void check_periodic_tables(bool all_directions)
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            std::array<bool, dim> periodic;
            periodic.fill(all_directions);
            periodic[0] = true;
            auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).periodic(periodic).disable_minimal_ghost_width();
            auto mesh = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a blob that crosses the boundary in x, so that fine cells have periodic images
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  const double c = d == 0 ? cell.center(0) - 0.05 : cell.center(d) - 0.5;
                                  r2 += c * c;
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - 0.2));
                          });
            if (!all_directions)
            {
                make_bc<Dirichlet<1>>(u, 0.);
            }
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            std::size_t n_copies = 0;
            ASSERT_FALSE(u.mesh().ghost_tables().empty());
            for (const auto& t : u.mesh().ghost_tables().periodic)
            {
                for (const auto& direction : t.directions)
                {
                    n_copies += direction.copies.size();
                }
            }
            ASSERT_GT(n_copies, 0u);

            auto s = make_scalar_field<double>("s", u.mesh());
            auto v = make_vector_field<double, 3>("v", u.mesh());
            fill_all(s, 0.1);
            fill_all(v, 0.7);

            for (std::size_t level = 0; level <= u.mesh().max_level(); ++level)
            {
                auto s_sets  = s;
                auto v_sets  = v;
                auto s_table = s;
                auto v_table = v;
                copy_periodic_with_sets(level, s_sets);
                copy_periodic_with_sets(level, v_sets);
                update_ghost_periodic(level, s_table, v_table);
                EXPECT_TRUE(same_bits(s_sets, s_table) && same_bits(v_sets, v_table)) << "periodic ghosts at level " << level;
            }
        }

        // project_bc, predict_bc and the corner copies through the tables of the mesh
        // and through the set expressions.
        template <std::size_t dim>
        void check_outer_tables()
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config                 = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(4);
            auto mesh                   = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a front across x = 0.5 that cuts the other boundaries: the levels change
            // along them, so predict_bc has ghosts to fill
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              u[cell] = std::tanh(100 * (cell.center(0) - 0.5));
                          });
            make_bc<Neumann<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            const auto& outer = u.mesh().ghost_tables().outer;
            ASSERT_FALSE(outer.empty());
            std::size_t n_projected = 0;
            std::size_t n_predicted = 0;
            std::size_t n_corners   = 0;
            for (const auto& level_tables : outer)
            {
                for (const auto& by_layer : level_tables.projection)
                {
                    for (const auto& t : by_layer)
                    {
                        n_projected += t.dest.size();
                    }
                }
                for (const auto& by_layer : level_tables.prediction)
                {
                    for (const auto& t : by_layer)
                    {
                        n_predicted += t.dest.size();
                    }
                }
                n_corners += level_tables.corners.dest.size();
            }
            ASSERT_GT(n_projected, 0u);
            // in 1D the boundary is a single cell: no level jump along it to predict
            if constexpr (dim > 1)
            {
                ASSERT_GT(n_predicted, 0u);
                ASSERT_GT(n_corners, 0u);
            }

            auto s = make_scalar_field<double>("s", u.mesh());
            auto v = make_vector_field<double, 3>("v", u.mesh());
            make_bc<Dirichlet<2>>(s, 0.);
            make_bc<Dirichlet<2>>(v, 0., 0., 0.);
            fill_all(s, 0.1);
            fill_all(v, 0.7);
            const auto& m = u.mesh();

            auto check = [&](auto& f, const std::string& name)
            {
                for (std::size_t level = m.min_level() - 1; level <= m.max_level(); ++level)
                {
                    for_each_cartesian_direction<dim>(
                        [&](auto, const auto& direction)
                        {
                            if (level < m.max_level())
                            {
                                auto with_sets  = f;
                                auto with_table = f;
                                detail::with_bc_projection_ghosts(m,
                                                                  level,
                                                                  direction,
                                                                  1,
                                                                  [&](auto& ghosts)
                                                                  {
                                                                      project_bc(ghosts, level, direction, 1, with_sets);
                                                                  });
                                project_bc(level, direction, 1, with_table);
                                EXPECT_TRUE(same_bits(with_sets, with_table)) << name << ": project_bc to level " << level;
                            }
                            if (level > m.min_level())
                            {
                                auto with_sets  = f;
                                auto with_table = f;
                                for (int layer = 1; layer <= 2; ++layer)
                                {
                                    detail::with_bc_prediction_layer(m,
                                                                     level,
                                                                     direction,
                                                                     layer,
                                                                     [&](auto& ghosts)
                                                                     {
                                                                         predict_bc(ghosts, level, direction, with_sets);
                                                                     });
                                }
                                predict_bc(level, direction, with_table);
                                EXPECT_TRUE(same_bits(with_sets, with_table)) << name << ": predict_bc at level " << level;
                            }
                        });
                    if constexpr (dim > 1)
                    {
                        if (level >= m.min_level())
                        {
                            auto with_sets  = f;
                            auto with_table = f;
                            for_each_diagonal_direction<dim>(
                                [&](const auto& direction)
                                {
                                    project_corner_below(level, direction, with_sets);
                                });
                            project_outer_corners_below(level, with_table);
                            EXPECT_TRUE(same_bits(with_sets, with_table)) << name << ": corner copies from level " << level;
                        }
                    }
                }
            };
            check(s, "scalar");
            check(v, "vector");
        }

        // update_further_ghosts_by_polynomial_extrapolation and
        // update_outer_corners_by_polynomial_extrapolation through the tables of
        // the mesh and through the set expressions. A ghost width of 3 and a B.C.
        // that fills one layer use the stencils of size 2, 4 and 6.
        template <std::size_t dim>
        void check_extrapolation_tables()
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config                 = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(6);
            auto mesh                   = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              u[cell] = std::tanh(100 * (cell.center(0) - 0.5));
                          });
            make_bc<Neumann<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));
            const auto& m = u.mesh();
            ASSERT_EQ(m.ghost_width(), 3);

            std::array<std::size_t, 7> n_stencils{}; // by stencil size
            for (const auto& level_tables : m.ghost_tables().outer)
            {
                auto count = [&](const auto& sequence)
                {
                    for (const auto& t : sequence)
                    {
                        n_stencils[t.stencil_size] += t.cells.size() / t.stencil_size;
                    }
                };
                for (const auto& by_layer : level_tables.extrapolation_on_ghosts)
                {
                    for (const auto& sequence : by_layer)
                    {
                        count(sequence);
                    }
                }
                for (const auto& corner : level_tables.corner_extrapolation)
                {
                    count(corner.sequence);
                }
            }
            // in 1D there is neither a corner nor a ghost below the projected ones: only the
            // layers extrapolated from the cells, recorded on first use and checked below
            if constexpr (dim > 1)
            {
                ASSERT_GT(n_stencils[2], 0u); // corners
                ASSERT_GT(n_stencils[4], 0u);
                ASSERT_GT(n_stencils[6], 0u);
            }

            auto s = make_scalar_field<double>("s", m);
            auto v = make_vector_field<double, 3>("v", m);
            make_bc<Dirichlet<1>>(s, 0.);
            make_bc<Dirichlet<1>>(v, 0., 0., 0.);
            fill_all(s, 0.1);
            fill_all(v, 0.7);

            auto check = [&](auto& f, const std::string& name)
            {
                using field_t = std::decay_t<decltype(f)>;
                for (std::size_t level = m.min_level(); level <= m.max_level(); ++level)
                {
                    for_each_cartesian_direction<dim>(
                        [&](auto, const auto& direction)
                        {
                            auto with_sets  = f;
                            auto with_table = f;
                            detail::FieldExtrapolation<field_t> visitor{with_sets};
                            detail::visit_extrapolation_on_cells(m, level, direction, 2, 3, visitor);
                            detail::visit_extrapolation_on_ghosts(m, level, direction, 2, 3, visitor);
                            update_further_ghosts_by_polynomial_extrapolation(level, direction, with_table);
                            EXPECT_TRUE(same_bits(with_sets, with_table)) << name << ": further ghosts at level " << level;
                        });
                    if constexpr (dim > 1)
                    {
                        for_each_diagonal_direction<dim>(
                            [&](const auto& direction)
                            {
                                auto with_sets  = f;
                                auto with_table = f;
                                detail::FieldExtrapolation<field_t> visitor{with_sets};
                                detail::visit_corner_extrapolation(m, level, direction, visitor);
                                update_outer_corners_by_polynomial_extrapolation(level, direction, with_table);
                                EXPECT_TRUE(same_bits(with_sets, with_table)) << name << ": corners at level " << level;
                            });
                    }
                }
            };
            check(s, "scalar");
            check(v, "vector");

            // The layers extrapolated from the cells are recorded the first time a field
            // asks for them: the B.C. above fills one layer, so the layers 2 and 3 now are.
            std::size_t n_recorded = 0;
            for (const auto& level_tables : m.ghost_tables().outer)
            {
                for (const auto& layers : level_tables.extrapolation_on_cells)
                {
                    for (int layer = 2; layer <= static_cast<int>(layers.size()); ++layer)
                    {
                        for (const auto& t : layers.get(layer,
                                                        [](auto&)
                                                        {
                                                            FAIL() << "a layer extrapolated from the cells was not recorded";
                                                        }))
                        {
                            n_recorded += t.cells.size();
                        }
                    }
                }
            }
            EXPECT_GT(n_recorded, 0u);
        }
    }

    TEST(ghost_tables, same_values_as_sets_1d)
    {
        check_tables<1>();
    }

    TEST(ghost_tables, same_values_as_sets_2d)
    {
        check_tables<2>();
    }

    TEST(ghost_tables, same_values_as_sets_3d)
    {
        check_tables<3>();
    }
}

namespace samurai
{
    TEST(ghost_tables, same_periodic_copies_as_sets_2d)
    {
        check_periodic_tables<2>(true);
        check_periodic_tables<2>(false);
    }

    TEST(ghost_tables, same_periodic_copies_as_sets_3d)
    {
        check_periodic_tables<3>(true);
        check_periodic_tables<3>(false);
    }
}

namespace samurai
{
    TEST(ghost_tables, same_outer_ghosts_as_sets_1d)
    {
        check_outer_tables<1>();
    }

    TEST(ghost_tables, same_outer_ghosts_as_sets_2d)
    {
        check_outer_tables<2>();
    }

    TEST(ghost_tables, same_outer_ghosts_as_sets_3d)
    {
        check_outer_tables<3>();
    }
}

namespace samurai
{
    TEST(ghost_tables, same_extrapolation_as_sets_1d)
    {
        check_extrapolation_tables<1>();
    }

    TEST(ghost_tables, same_extrapolation_as_sets_2d)
    {
        check_extrapolation_tables<2>();
    }

    TEST(ghost_tables, same_extrapolation_as_sets_3d)
    {
        check_extrapolation_tables<3>();
    }
}
