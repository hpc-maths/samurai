// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The tables of update_ghost_mr against the set expressions they replace: on an
// adapted mesh, the field projection, the prediction, the periodic copies and the
// ghosts outside the domain filled from other ghosts (project_bc, predict_bc and
// the corner copies) or by polynomial extrapolation must give the same values, bit for bit, at every level, for a
// scalar and a vector field. The tables of the ghosts outside the domain, built
// from direct walks of the mesh, must also hold the very entries the set
// expressions give, in the same order.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

namespace samurai
{
    namespace
    {
        // The tables are local to a rank, and a rank may hold no boundary: the
        // counts that show a test is not empty are summed over the ranks.
        std::size_t over_ranks(std::size_t n)
        {
#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            return boost::mpi::all_reduce(world, n, std::plus<std::size_t>());
#else
            return n;
#endif
        }

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
            ASSERT_GT(over_ranks(n_projected), 0u);
            ASSERT_GT(over_ranks(n_predicted), 0u);

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
            ASSERT_GT(over_ranks(n_copies), 0u);

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
            ASSERT_GT(over_ranks(n_projected), 0u);
            // in 1D the boundary is a single cell: no level jump along it to predict
            if constexpr (dim > 1)
            {
                ASSERT_GT(over_ranks(n_predicted), 0u);
                ASSERT_GT(over_ranks(n_corners), 0u);
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
                ASSERT_GT(over_ranks(n_stencils[2]), 0u); // corners
                ASSERT_GT(over_ranks(n_stencils[4]), 0u);
                ASSERT_GT(over_ranks(n_stencils[6]), 0u);
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
            EXPECT_GT(over_ranks(n_recorded), 0u);
        }
    }

    namespace
    {
        bool same_sequence(const detail::ExtrapolationSequence& a, const detail::ExtrapolationSequence& b)
        {
            if (a.size() != b.size())
            {
                return false;
            }
            for (std::size_t k = 0; k < a.size(); ++k)
            {
                if (a[k].stencil_size != b[k].stencil_size || a[k].cells != b[k].cells)
                {
                    return false;
                }
            }
            return true;
        }

        std::size_t n_stencil_cells(const detail::ExtrapolationSequence& sequence)
        {
            std::size_t n = 0;
            for (const auto& t : sequence)
            {
                n += t.cells.size();
            }
            return n;
        }

        // Entries of the outer ghost tables by kind, to make sure a mesh has some.
        struct OuterEntries
        {
            std::size_t projection    = 0;
            std::size_t prediction    = 0;
            std::size_t corners       = 0;
            std::size_t on_ghosts     = 0;
            std::size_t on_cells      = 0;
            std::size_t corner_layers = 0;
        };

        // The outer ghost tables of every level from the direct walks against
        // those from the set expressions, entry by entry and in order, the layers
        // extrapolated from the cells included. Each walk must apply: none may
        // fall back on the set expressions.
        template <class Mesh>
        OuterEntries expect_same_outer_tables(const Mesh& mesh, const std::string& name)
        {
            constexpr std::size_t dim = Mesh::dim;
            OuterEntries n;

            for (std::size_t level = 0; level <= mesh.max_level(); ++level)
            {
                const auto where = name + " at level " + std::to_string(level);
                const auto scans = detail::make_outer_ghost_tables(level, mesh);
                const auto sets  = detail::make_outer_ghost_tables(level, mesh, false);

                EXPECT_TRUE(detail::as_block(mesh.domain(level))) << where;

                EXPECT_EQ(scans.projection.size(), sets.projection.size()) << where;
                for (std::size_t d = 0; d < std::min(scans.projection.size(), sets.projection.size()); ++d)
                {
                    EXPECT_EQ(scans.projection[d].size(), sets.projection[d].size()) << where;
                    for (std::size_t l = 0; l < std::min(scans.projection[d].size(), sets.projection[d].size()); ++l)
                    {
                        const auto& a = scans.projection[d][l];
                        const auto& b = sets.projection[d][l];
                        EXPECT_TRUE(a.dest == b.dest && a.start == b.start && a.child == b.child)
                            << where << ": project_bc in direction " << d << ", layer " << l + 1;
                        n.projection += b.dest.size();
                    }
                }

                EXPECT_EQ(scans.prediction.size(), sets.prediction.size()) << where;
                for (std::size_t d = 0; d < std::min(scans.prediction.size(), sets.prediction.size()); ++d)
                {
                    EXPECT_EQ(scans.prediction[d].size(), sets.prediction[d].size()) << where;
                    for (std::size_t l = 0; l < std::min(scans.prediction[d].size(), sets.prediction[d].size()); ++l)
                    {
                        const auto& a = scans.prediction[d][l];
                        const auto& b = sets.prediction[d][l];
                        EXPECT_TRUE(a.dest == b.dest && a.src == b.src) << where << ": predict_bc in direction " << d << ", layer " << l + 1;
                        n.prediction += b.dest.size();
                    }
                }

                EXPECT_TRUE(scans.corners.dest == sets.corners.dest && scans.corners.src == sets.corners.src) << where << ": corner copies";
                n.corners += sets.corners.dest.size();

                EXPECT_EQ(scans.extrapolation_on_ghosts.size(), sets.extrapolation_on_ghosts.size()) << where;
                for (std::size_t d = 0; d < std::min(scans.extrapolation_on_ghosts.size(), sets.extrapolation_on_ghosts.size()); ++d)
                {
                    EXPECT_EQ(scans.extrapolation_on_ghosts[d].size(), sets.extrapolation_on_ghosts[d].size()) << where;
                    for (std::size_t l = 0; l < std::min(scans.extrapolation_on_ghosts[d].size(), sets.extrapolation_on_ghosts[d].size());
                         ++l)
                    {
                        EXPECT_TRUE(same_sequence(scans.extrapolation_on_ghosts[d][l], sets.extrapolation_on_ghosts[d][l]))
                            << where << ": extrapolation on the ghosts in direction " << d << ", layer " << l + 1;
                        n.on_ghosts += n_stencil_cells(sets.extrapolation_on_ghosts[d][l]);
                    }
                }

                EXPECT_EQ(scans.corner_extrapolation.size(), sets.corner_extrapolation.size()) << where;
                for (std::size_t c = 0; c < std::min(scans.corner_extrapolation.size(), sets.corner_extrapolation.size()); ++c)
                {
                    EXPECT_EQ(scans.corner_extrapolation[c].recorded, sets.corner_extrapolation[c].recorded) << where;
                    EXPECT_TRUE(same_sequence(scans.corner_extrapolation[c].sequence, sets.corner_extrapolation[c].sequence))
                        << where << ": corner extrapolation " << c;
                    n.corner_layers += n_stencil_cells(sets.corner_extrapolation[c].sequence);
                }

                // the walks that may fall back on the set expressions
                if (level >= mesh.min_level())
                {
                    const auto domain = detail::as_block(mesh.domain(level));
                    for_each_cartesian_direction<dim>(
                        [&](auto d, const auto& direction)
                        {
                            if (mesh.is_periodic(d))
                            {
                                return;
                            }
                            for (int layer = 1; layer <= static_cast<int>(mesh.ghost_width()); ++layer)
                            {
                                detail::ExtrapolationSequence scanned;
                                detail::ExtrapolationSequence visited;
                                EXPECT_TRUE(domain && detail::scan_extrapolation_on_cells(mesh, level, *domain, direction, layer, scanned))
                                    << where << ": extrapolation on the cells in direction " << d << ", layer " << layer;
                                detail::ExtrapolationRecorder recorder{visited};
                                detail::visit_extrapolation_on_cells(mesh, level, direction, layer, layer, recorder);
                                EXPECT_TRUE(same_sequence(scanned, visited))
                                    << where << ": extrapolation on the cells in direction " << d << ", layer " << layer;
                                n.on_cells += n_stencil_cells(visited);
                            }
                            if (level > mesh.min_level())
                            {
                                const auto coarse_domain = detail::as_block(mesh.domain(level - 1));
                                for (int layer = 1; layer <= mesh.max_stencil_radius(); ++layer)
                                {
                                    detail::CopyTable table;
                                    EXPECT_TRUE(coarse_domain && detail::scan_bc_prediction(mesh, level, *coarse_domain, direction, layer, table))
                                        << where << ": predict_bc in direction " << d << ", layer " << layer;
                                }
                            }
                        });
                    if constexpr (dim > 1)
                    {
                        for_each_diagonal_direction<dim>(
                            [&](const auto& direction)
                            {
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    if (direction[d] != 0 && mesh.is_periodic(d))
                                    {
                                        return;
                                    }
                                }
                                detail::ExtrapolationSequence sequence;
                                detail::CopyTable copies;
                                EXPECT_TRUE(detail::scan_corner_extrapolation(mesh, level, direction, sequence)) << where;
                                EXPECT_TRUE(detail::scan_corner_copies(mesh, level, direction, copies)) << where;
                            });
                    }
                }
            }
            return n;
        }

        // A mesh adapted to a blob of radius `radius` centred at `center`.
        template <std::size_t dim, class Config>
        auto
        blob_mesh(const Box<double, dim>& box, const Config& config, const xt::xtensor_fixed<double, xt::xshape<dim>>& center, double radius)
        {
            auto mesh = mra::make_mesh(box, config);
            auto u    = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - center[d]) * (cell.center(d) - center[d]);
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - radius));
                          });
            make_bc<Neumann<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));
            return u.mesh();
        }

        template <std::size_t dim>
        void check_outer_tables_by_scan()
        {
            using point_t               = xt::xtensor_fixed<double, xt::xshape<dim>>;
            const std::size_t max_level = dim == 3 ? 5 : 7;
            const Box<double, dim> unit = {xt::zeros<double>({dim}), xt::ones<double>({dim})};
            point_t corner              = xt::ones<double>({dim});
            point_t centre              = 0.4 * xt::ones<double>({dim});
            OuterEntries total;
            auto add = [&](const OuterEntries& n)
            {
                total.projection += n.projection;
                total.prediction += n.prediction;
                total.corners += n.corners;
                total.on_ghosts += n.on_ghosts;
                total.on_cells += n.on_cells;
                total.corner_layers += n.corner_layers;
            };

            // a blob across a corner: the finest level reaches the boundary
            for (int stencil_size : {2, 4, 6})
            {
                auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(stencil_size);
                add(expect_same_outer_tables(blob_mesh(unit, config, corner, 0.3), "corner blob, stencil " + std::to_string(stencil_size)));
            }

            // a blob inside: the faces only have the coarse levels, min_level 1
            // (stencil 6 as the ink mesh of the lid-driven cavity demo)
            for (int stencil_size : {4, 6})
            {
                auto config = mesh_config<dim>().min_level(1).max_level(max_level).max_stencil_size(stencil_size);
                add(expect_same_outer_tables(blob_mesh(unit, config, centre, 0.2), "inner blob, stencil " + std::to_string(stencil_size)));
            }

            // a box that is not a cube, with negative coordinates, and a gap of
            // several levels between min_level and the finest cells
            {
                point_t lo  = -0.75 * xt::ones<double>({dim});
                point_t hi  = 0.5 * xt::ones<double>({dim});
                hi[0]       = 1.25;
                point_t c   = lo;
                c[0]        = 0.;
                auto config = mesh_config<dim>().min_level(3).max_level(max_level).max_stencil_size(4).disable_minimal_ghost_width();
                add(expect_same_outer_tables(blob_mesh(Box<double, dim>(lo, hi), config, c, 0.25), "shifted box"));
            }

            // periodic in x: no table in x, nor at the corners it closes
            if constexpr (dim > 1)
            {
                std::array<bool, dim> periodic{};
                periodic[0] = true;
                auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(4).periodic(periodic);
                add(expect_same_outer_tables(blob_mesh(unit, config, corner, 0.3), "periodic in x"));
            }

            // a uniform mesh
            {
                auto config = mesh_config<dim>().min_level(3).max_level(3).max_stencil_size(4);
                add(expect_same_outer_tables(mra::make_mesh(unit, config), "uniform"));
            }

            EXPECT_GT(over_ranks(total.projection), 0u);
            EXPECT_GT(over_ranks(total.on_cells), 0u);
            if constexpr (dim > 1)
            {
                EXPECT_GT(over_ranks(total.prediction), 0u);
                EXPECT_GT(over_ranks(total.corners), 0u);
                EXPECT_GT(over_ranks(total.on_ghosts), 0u);
                EXPECT_GT(over_ranks(total.corner_layers), 0u);
            }
        }
    }

    TEST(ghost_tables, outer_tables_by_scan_1d)
    {
        check_outer_tables_by_scan<1>();
    }

    TEST(ghost_tables, outer_tables_by_scan_2d)
    {
        check_outer_tables_by_scan<2>();
    }

    TEST(ghost_tables, outer_tables_by_scan_3d)
    {
        check_outer_tables_by_scan<3>();
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
