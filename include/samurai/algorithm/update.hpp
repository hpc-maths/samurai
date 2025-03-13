// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>

#include <xtensor/xfixed.hpp>

#include "../algorithm.hpp"
#include "../bc.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../subset/node.hpp"
#include "../timers.hpp"
#include "graduation.hpp"
#include "utils.hpp"

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <xtensor/xmasked_view.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <class Field, class... Fields>
    void update_ghost(Field& field, Fields&... fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        auto& mesh            = field.mesh();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::proj_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, fields...));
        }

        update_bc(0, field, fields...);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            auto set_at_level = intersection(mesh[mesh_id_t::pred_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level);
            set_at_level.apply_op(variadic_prediction<pred_order, false>(field, fields...));
            update_bc(level, field, fields...);
        }
    }

    template <class Field>
    void update_ghost_mro(Field& field)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;
        auto& mesh                       = field.mesh();

        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(projection(field));
        }

        update_bc(0, field);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            // auto expr =
            // difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
            //                                                union_(mesh[mesh_id_t::cells][level],
            //                                                       mesh[mesh_id_t::proj_cells][level])),
            //                                     mesh.domain()),
            //                        difference(mesh[mesh_id_t::overleaves][level],
            //                                   union_(mesh[mesh_id_t::union_cells][level],
            //                                          mesh[mesh_id_t::cells_and_ghosts][level])))
            //             .on(level);

            auto expr = intersection(
                difference(mesh[mesh_id_t::all_cells][level], union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                self(mesh.domain()).on(level));

            expr.apply_op(prediction<pred_order, false>(field));
            update_bc(level, field);
        }
    }

    template <class Field, class... Fields>
    void project_bc(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field, Fields&... other_fields)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        constexpr std::size_t dim        = Field::dim;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        static_assert(pred_order == 1);

        auto& mesh              = field.mesh();
        auto domain             = self(mesh.domain()).on(level);
        auto& cells             = mesh[mesh_id_t::cells][level];
        auto& all               = mesh[mesh_id_t::reference][level];
        auto& inner_proj_ghosts = mesh[mesh_id_t::proj_cells][level];

        auto d                  = find_direction_index(direction);
        bool positive_direction = direction[d] == 1;

        std::cout << "================================================= direction = " << direction << ", level = " << level << std::endl;

        for (std::size_t lower = 1; lower <= 1; ++lower) // lower level
        {
            auto proj_level = level - lower;
            for (int layer_width = 1; layer_width <= 1; layer_width++) // number of fine cell layers to project
            {
                // auto possible_outside_layer = difference(translate(cells, layer_width * direction), domain);
                // auto outside_layer          = intersection(possible_outside_layer, mesh[mesh_id_t::reference][level]);

                //-- with cells
                // auto outside_layer = difference(translate(cells, layer_width * direction), domain);

                //-- with all
                auto inner_all     = layer_width == 1 ? intersection(union_(inner_proj_ghosts, cells), domain)
                                                      : intersection(union_(cells, cells), cells);
                auto outside_layer = difference(translate(inner_all, layer_width * direction), domain);

                auto projection_ghosts = intersection(outside_layer, mesh[mesh_id_t::reference][proj_level]).on(proj_level);

                // Projection of 2^(dim-1) children
                std::size_t n_children           = static_cast<std::size_t>(layer_width) * (1 << lower); // layer_width * 2^(dim-1)
                std::size_t n_children_per_layer = n_children / static_cast<std::size_t>(layer_width);
                double avg_factor                = 1. / static_cast<double>(n_children);

                std::cout << "\tlower = " << lower << ", layer_width = " << layer_width << ", n_children = " << n_children << std::endl;
                if (lower == 3)
                {
                    // std::cout << "n_children = " << n_children << std::endl;
                    // std::cout << "avg_factor = " << avg_factor << std::endl;

                    std::cout << "outside_layer = " << std::endl;
                    outside_layer(
                        [&](const auto& i, const auto& index)
                        {
                            std::cout << "i = " << i << ", index = " << index << std::endl;
                        });
                }

                if constexpr (dim == 1)
                {
                    // Projection of 1 child
                    if (positive_direction) // right
                    {
                        projection_ghosts(
                            [&](const auto& i, const auto&)
                            {
                                auto i_child         = layer_width * (1 << lower) * i; // 2i if lower == 1, 4i if lower == 2
                                field(proj_level, i) = field(level, i_child);
                            });
                    }
                    else // left
                    {
                        projection_ghosts(
                            [&](const auto& i, const auto&)
                            {
                                auto i_child         = layer_width * (1 << lower) * i;        // (2i+1) if lower == 1, 4i if lower == 2
                                field(proj_level, i) = field(level, layer_width * 2 * i + 1); // 2i+1 fixed
                            });
                    }
                }
                else
                {
                    // Projection of 2 children
                    if (d == 0) // x-direction
                    {
                        projection_ghosts(
                            [&](const auto& i, const auto& index)
                            {
                                using index_t = std::decay_t<decltype(index)>;

                                std::cout << "--------------------------------" << std::endl;
                                std::cout << "field(" << proj_level << "," << i << "," << index << ") = " << field(proj_level, i, index)
                                          << std::endl;

                                field(proj_level, i, index) = 0;

                                // if positive direction, right --> 2i   fixed
                                //                   else left  --> 2i+1 fixed
                                auto i_child = (1 << lower) * i;
                                if (!positive_direction)
                                {
                                    i_child.start += (1 << lower) - 1;
                                }
                                i_child.end = i_child.start + 1;
                                // i_child.step = 1;

                                index_t index_child = (1 << lower) * index;

                                for (std::size_t layer = 0; layer < static_cast<std::size_t>(layer_width); ++layer)
                                {
                                    for (std::size_t d1 = 0; d1 < dim - 1; ++d1) // next direction
                                    {
                                        for (std::size_t dummy1 = 0; dummy1 < n_children_per_layer; ++dummy1, ++index_child[d1])
                                        {
                                            if constexpr (dim == 2)
                                            {
                                                std::cout << "field(" << proj_level << "," << i << "," << index << ") += " << avg_factor
                                                          << " * field(" << level << "," << i_child << "," << index_child << ")" << " = "
                                                          << field(level, i_child, index_child) << std::endl;
                                                if (xt::any(xt::isnan(field(level, i_child, index_child))))
                                                {
                                                    std::cout << "NaN in field(" << level << "," << i_child << "," << index_child << ")"
                                                              << std::endl;
                                                    // std::exit(1);
                                                }
                                                field(proj_level, i, index) += avg_factor * field(level, i_child, index_child);
                                            }
                                            else if constexpr (dim == 3)
                                            {
                                                for (std::size_t d2 = d1 + 1; d2 < dim - 1; ++d2)
                                                {
                                                    for (std::size_t dummy2 = 0; dummy2 < n_children_per_layer; ++dummy2, ++index_child[d2])
                                                    {
                                                        field(proj_level, i, index) += avg_factor * field(level, i_child, index_child);
                                                    }
                                                    index_child[d2] -= n_children_per_layer; // reset index_child[d2]
                                                }
                                            }
                                        }
                                        index_child[d1] -= n_children_per_layer; // reset index_child[d1]
                                    }
                                    i_child += positive_direction ? 1 : -1; // next layer
                                }
                            });
                    }
                    else // other directions
                    {
                        projection_ghosts(
                            [&](const auto& i, const auto& index)
                            {
                                using index_t = std::decay_t<decltype(index)>;

                                field(proj_level, i, index) = 0;

                                // std::cout << "--------------------------------" << std::endl;
                                // std::cout << "field(" << proj_level << "," << i << "," << index << ") = " << field(proj_level, i, index)
                                //           << std::endl;

                                // if positive direction, top     --> 2j   fixed (=index[d-1])
                                //                   else bottom  --> 2j+1 fixed
                                auto i_child        = (1 << lower) * i;
                                index_t index_child = (1 << lower) * index;
                                // i_child.step        = 1;
                                if (!positive_direction)
                                {
                                    index_child[d - 1] += (1 << lower) - 1;
                                }

                                for (std::size_t layer = 0; layer < static_cast<std::size_t>(layer_width); ++layer)
                                {
                                    for (std::size_t dummy1 = 0; dummy1 < n_children_per_layer; ++dummy1, i_child += 1) // i loop
                                    {
                                        if constexpr (dim == 2)
                                        {
                                            // std::cout << "field(" << proj_level << "," << i << "," << index << ") += " << avg_factor
                                            //           << " * field(" << level << "," << i_child << "," << index_child << ")" << " = "
                                            //           << field(level, i_child, index_child) << std::endl;
                                            if (xt::any(xt::isnan(field(level, i_child, index_child))))
                                            {
                                                std::cout << "NaN in field(" << level << "," << i_child << "," << index_child << ")"
                                                          << std::endl;
                                                // std::exit(1);
                                            }
                                            field(proj_level, i, index) = avg_factor * field(level, i_child, index_child);
                                        }
                                        else if constexpr (dim == 3)
                                        {
                                            for (std::size_t d2 = 0; d2 < dim - 1; ++d2)
                                            {
                                                if (d2 != d - 1)
                                                {
                                                    for (std::size_t dummy2 = 0; dummy2 < n_children_per_layer; ++dummy2, ++index_child[d2])
                                                    {
                                                        field(proj_level, i, index) += avg_factor * field(level, i_child, index_child);
                                                    }
                                                    index_child[d2] -= n_children_per_layer; // reset index_child[d2]
                                                }
                                            }
                                        }
                                    }
                                    i_child -= static_cast<int>(n_children_per_layer); // reset i_child
                                    index_child[d - 1] += positive_direction ? 1 : -1; // next layer
                                }
                            });
                    }
                }
            }
        }
    }

    template <class Field>
    void project_corner(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        for (std::size_t lower = 1; lower <= 2; ++lower) // lower level (1 or 2)
        {
            // std::cout << "lower = " << lower << std::endl;
            auto proj_level = level - lower;

            auto fine_inner_corner = get_corner(mesh, level, direction);
            auto fine_outer_corner = intersection(translate(fine_inner_corner, direction), mesh[mesh_id_t::reference][level]);
            auto projection_ghost  = intersection(fine_outer_corner.on(proj_level), mesh[mesh_id_t::reference][proj_level]);

            projection_ghost(
                [&](const auto& i, const auto& index)
                {
                    using index_t = std::decay_t<decltype(index)>;

                    // std::cout << "-------------------" << std::endl;

                    auto i_child = (1 << lower) * i;
                    i_child.start += direction[0] == -1 ? ((1 << lower) - 1) : 0;
                    i_child.end         = i_child.start + 1;
                    i_child.step        = 1;
                    index_t index_child = (1 << lower) * index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index_child[d] += direction[d + 1] == -1 ? ((1 << lower) - 1) : 0;
                    }
                    // std::cout << "field(" << proj_level << "," << i << "," << index << ") = field(" << level << "," << i_child << ","
                    //           << index_child << ")"
                    //           << " = ";
                    // std::cout << field(level, i_child, index_child) << std::endl;
                    field(proj_level, i, index) = field(level, i_child, index_child);
                });
        }
    }

    template <class Field, class... Fields>
    void apply_bc_and_project_below(std::size_t level, Field& field, Fields&... other_fields)
    {
        static constexpr std::size_t dim = Field::dim;

        // Corners
        static_nested_loop<dim, -1, 2>(
            [&](auto& direction)
            {
                int number_of_ones = xt::sum(xt::abs(direction))[0];
                if (number_of_ones > 1) // corner
                {
                    update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                    // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                    project_corner(level, direction, field, other_fields...);
                    // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                }
            });

        DirectionVector<dim> direction;
        direction.fill(0);

        // Apply the B.C. at the same level as the cells
        for (std::size_t d = 0; d < dim; ++d) // for all Cartesian direction
        {
            direction[d] = 1;
            update_bc_for_scheme(level, direction, field, other_fields...);
            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
            direction[d] = -1;
            update_bc_for_scheme(level, direction, field, other_fields...);
            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
            direction[d] = 0;
        }

        // Project the B.C. onto the two levels below
        for (std::size_t d = 0; d < dim; ++d) // for all Cartesian direction
        {
            direction[d] = 1;
            project_bc(level, direction, field, other_fields...);
            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
            direction[d] = -1;
            project_bc(level, direction, field, other_fields...);
            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
            direction[d] = 0;
        }
    }

    template <class Field, class... Fields>
    void update_ghost_mr(Field& field, Fields&... other_fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t dim        = Field::dim;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        times::timers.start("ghost update");

        auto& mesh            = field.mesh();
        auto max_level        = mesh.max_level();
        std::size_t min_level = 0;

        for (std::size_t level = max_level; level > min_level; --level)
        {
            update_ghost_periodic(level, field, other_fields...);
            update_ghost_subdomains(level, field, other_fields...);

            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, other_fields...));
        }

        // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, mesh, field);

        for (std::size_t level = mesh.max_level(); level >= mesh.min_level() - 2; --level)
        {
            // apply_bc_and_project_below(level, field, other_fields...);
            if (level >= mesh.min_level())
            {
                //  Corners
                static_nested_loop<dim, -1, 2>(
                    [&](auto& direction)
                    {
                        int number_of_ones = xt::sum(xt::abs(direction))[0];
                        if (number_of_ones > 1) // corner
                        {
                            update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                            project_corner(level, direction, field, other_fields...);
                            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                        }
                    });
            }

            samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);

            DirectionVector<dim> direction;
            direction.fill(0);

            if (level < mesh.max_level())
            {
                // Project the B.C. from level+1 to level
                for (std::size_t d = 0; d < dim; ++d) // for all Cartesian direction
                {
                    direction[d] = 1;
                    project_bc(level + 1, direction, field, other_fields...);
                    // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                    direction[d] = -1;
                    project_bc(level + 1, direction, field, other_fields...);
                    samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                    direction[d] = 0;
                }
            }
            if (level >= mesh.min_level())
            {
                // Apply the B.C. at the same level as the cells
                for (std::size_t d = 0; d < dim; ++d) // for all Cartesian direction
                {
                    direction[d] = 1;
                    update_bc_for_scheme(level, direction, field, other_fields...);
                    // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                    direction[d] = -1;
                    update_bc_for_scheme(level, direction, field, other_fields...);
                    // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, field.mesh(), field);
                    direction[d] = 0;
                }
            }
        }

        // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, mesh, field);

        if (min_level > 0 && min_level != max_level)
        {
            // update_bc(min_level - 1, field, other_fields...);
            update_ghost_periodic(min_level - 1, field, other_fields...);
            update_ghost_subdomains(min_level - 1, field, other_fields...);
        }
        // update_bc(min_level, field, other_fields...);
        update_ghost_periodic(min_level, field, other_fields...);
        update_ghost_subdomains(min_level, field, other_fields...);

        // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, mesh, field);

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto expr = intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                                     mesh.subdomain(),
                                     mesh[mesh_id_t::all_cells][level - 1])
                            .on(level);

            expr.apply_op(variadic_prediction<pred_order, false>(field, other_fields...));
            update_ghost_periodic(level, field, other_fields...);
            update_ghost_subdomains(level, field, other_fields...);
            // samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, mesh, field);
            //  update_bc(level, field, other_fields...);
        }
        samurai::save(fs::current_path(), fmt::format("update_ghosts"), {true, true}, mesh, field);

        times::timers.stop("ghost update");
    }

    inline void update_ghost_mr()
    {
    }

    template <class... T>
    inline void update_ghost_mr(std::tuple<T...>& fields)
    {
        std::apply(
            [](T&... tupleArgs)
            {
                update_ghost_mr(tupleArgs...);
            },
            fields);
    }

    template <class... T>
    inline void update_ghost_mr(Field_tuple<T...>& fields)
    {
        update_ghost_mr(fields.elements());
    }

    template <class Field>
    void update_ghost_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        // static constexpr std::size_t dim = Field::dim;
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh = field.mesh();
        mpi::communicator world;
        std::vector<std::vector<value_t>> to_send(mesh.mpi_neighbourhood().size());

        std::size_t i_neigh = 0;
        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                  neighbour.mesh[mesh_id_t::reference][level],
                                                  mesh.subdomain())
                                         .on(level);
                out_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(field(level, i, index).begin(), field(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });

                req.push_back(world.isend(neighbour.rank, neighbour.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                std::vector<value_t> to_recv;
                std::ptrdiff_t count = 0;

                world.recv(neighbour.rank, world.rank(), to_recv);
                auto in_interface = intersection(neighbour.mesh[mesh_id_t::reference][level],
                                                 mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh.subdomain())
                                        .on(level);
                in_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(to_recv.begin() + count,
                                  to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * Field::n_comp),
                                  field(level, i, index).begin());
                        count += static_cast<ptrdiff_t>(i.size() * Field::n_comp);
                    });
            }
        }
        mpi::wait_all(req.begin(), req.end());
#endif
    }

    template <class Field, class... Fields>
    void update_ghost_subdomains(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_subdomains(level, field);
        update_ghost_subdomains(level, other_fields...);
    }

    template <class Field>
    void update_ghost_subdomains([[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto& mesh     = field.mesh();
        auto max_level = mesh.max_level();
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            update_ghost_subdomains(level, field);
        }
#endif
    }

    template <class Field>
    void update_tag_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& tag, [[maybe_unused]] bool erase = false)
    {
#ifdef SAMURAI_WITH_MPI
        //  constexpr std::size_t dim = Field::dim;
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh = tag.mesh();
        mpi::communicator world;
        std::vector<std::vector<value_t>> to_send(mesh.mpi_neighbourhood().size());

        std::size_t i_neigh = 0;
        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                  neighbour.mesh[mesh_id_t::reference][level],
                                                  mesh.subdomain())
                                         .on(level);
                out_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });

                req.push_back(world.isend(neighbour.rank, neighbour.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                std::vector<value_t> to_recv;
                std::ptrdiff_t count = 0;

                world.recv(neighbour.rank, world.rank(), to_recv);

                auto in_interface = intersection(mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh.subdomain())
                                        .on(level);
                in_interface(
                    [&](const auto& i, const auto& index)
                    {
                        xt::xtensor<value_t, 1> neigh_tag = xt::empty_like(tag(level, i, index));
                        std::copy(to_recv.begin() + count, to_recv.begin() + count + static_cast<std::ptrdiff_t>(i.size()), neigh_tag.begin());
                        if (erase)
                        {
                            tag(level, i, index) = neigh_tag;
                        }
                        else
                        {
                            tag(level, i, index) |= neigh_tag;
                        }
                        count += static_cast<std::ptrdiff_t>(i.size());
                    });
            }
        }
        mpi::wait_all(req.begin(), req.end());

#endif
    }

    template <class Field>
    void check_duplicate_cells([[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        // static constexpr std::size_t dim = Field::dim;
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh[mesh_id_t::cells][level]);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            // delete cell
                            std::cout << fmt::format("fall intersection between {} {} on level {} in {} {}",
                                                     world.rank(),
                                                     neighbour.rank,
                                                     level,
                                                     i,
                                                     index[0])
                                      << std::endl;
                        });
                }
            }
        }
#endif
    }

    template <class Field>
    void keep_only_one_coarse_tag([[maybe_unused]] Field& tag)
    {
#ifdef SAMURAI_WITH_MPI
        constexpr std::size_t dim = Field::dim;
        using mesh_t              = typename Field::mesh_t;
        using mesh_id_t           = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = tag.mesh();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                auto min_level = std::max<std::size_t>(1, mesh[mesh_id_t::reference].min_level());

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh.subdomain()).on(level - 1);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            if constexpr (dim == 1)
                            {
                                auto mask1 = (tag(level, 2 * i) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1), mask) = 0;
                            }
                            if constexpr (dim == 2)
                            {
                                auto j     = index[0];
                                auto mask1 = (tag(level, 2 * i, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) = 0;
                            }
                            if constexpr (dim == 3)
                            {
                                auto j     = index[0];
                                auto k     = index[1];
                                auto mask1 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k), mask)             = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k + 1), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = 0;
                            }
                        });
                }
            }
        }
#endif
    }

    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        using field_value_t = typename Field::value_type;
#endif // SAMURAI_WITH_MPI
        using mesh_id_t        = typename Field::mesh_t::mesh_id_t;
        using config           = typename Field::mesh_t::config;
        using lca_type         = typename Field::mesh_t::lca_type;
        using interval_value_t = typename Field::interval_t::value_t;
        using box_t            = Box<interval_value_t, Field::dim>;

        constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        const auto& domain      = mesh.domain();
        const auto& min_indices = domain.min_indices();
        const auto& max_indices = domain.max_indices();

        const auto& mesh_ref = mesh[mesh_id_t::reference];

        const std::size_t delta_l = domain.level() - level;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;

        for (std::size_t d = 0; d < dim; ++d)
        {
            min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
            max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            shift[d]      = 0;
        }
#ifdef SAMURAI_WITH_MPI
        std::vector<mpi::request> req;
        req.reserve(mesh.mpi_neighbourhood().size());
        mpi::communicator world;

        std::vector<std::vector<field_value_t>> field_data_out(mesh.mpi_neighbourhood().size());
        std::vector<field_value_t> field_data_in;
#endif // SAMURAI_WITH_MPI
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        field(level, i, index) = field(level, i - shift_interval, index - shift_index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        field(level, i, index) = field(level, i + shift_interval, index + shift_index);
                    });
#ifdef SAMURAI_WITH_MPI
                size_t neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    field_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& field_data = field(level, i - shift_interval, index - shift_index);
                            std::copy(field_data.begin(), field_data.end(), std::back_inserter(field_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& field_data = field(level, i + shift_interval, index + shift_index);
                            std::copy(field_data.begin(), field_data.end(), std::back_inserter(field_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, field_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    world.recv(mpi_neighbor.rank, world.rank(), field_data_in);
                    auto it       = field_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(it, it + std::ssize(field(level, i, index)), field(level, i, index).begin());
                            it += std::ssize(field(level, i, index));
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(it, it + std::ssize(field(level, i, index)), field(level, i, index).begin());
                            it += std::ssize(field(level, i, index));
                        });
                }
                mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
                /* reset variables for next iterations. */
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            }
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(level, field);
        update_ghost_periodic(level, other_fields...);
    }

    template <class Field>
    void update_ghost_periodic(Field& field)
    {
        using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::reference].min_level();
        std::size_t max_level = mesh[mesh_id_t::reference].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_ghost_periodic(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(field);
        update_ghost_periodic(other_fields...);
    }

    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
        using mesh_id_t           = typename Tag::mesh_t::mesh_id_t;
        using config              = typename Tag::mesh_t::config;
        using lca_type            = typename Tag::mesh_t::lca_type;
        using interval_value_t    = typename Tag::interval_t::value_t;
        using box_t               = Box<interval_value_t, Tag::dim>;
        constexpr std::size_t dim = Tag::dim;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;

        auto& mesh           = tag.mesh();
        const auto& mesh_ref = mesh[mesh_id_t::reference];

        auto& domain     = mesh.domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        const std::size_t delta_l = domain.level() - level;

        for (std::size_t d = 0; d < dim; ++d)
        {
            shift[d]      = 0;
            min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
            max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
        }
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
        std::vector<mpi::request> req;
        req.reserve(mesh.mpi_neighbourhood().size());
        mpi::communicator world;

        std::vector<std::vector<tag_value_type>> tag_data_out(mesh.mpi_neighbourhood().size());
        std::vector<tag_value_type> tag_data_in;
#endif // SAMURAI_WITH_MPI
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                        tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i + shift_interval, index + shift_index);
                        tag(level, i + shift_interval, index + shift_index) |= tag(level, i, index);
                    });
#ifdef SAMURAI_WITH_MPI
                // first  pass tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                size_t neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i - shift_interval, index - shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i + shift_interval, index + shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
                // second pass tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i - shift_interval, index - shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i + shift_interval, index + shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
                /* reset variables for next iterations. */
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            }
        }
    }

    template <class Field>
    void update_overleaves_mr(Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        update_bc(min_level, field);
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level], mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction<1, false>(field));
            update_bc(level, field);
        }
    }

    namespace detail
    {
        template <class Mesh, class Field>
        void update_fields(Mesh& new_mesh, Field& field)
        {
            using mesh_id_t                  = typename Mesh::mesh_id_t;
            constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

            Field new_field("new_f", new_mesh);
#ifdef SAMURAI_CHECK_NAN
            new_field.fill(std::nan(""));
#else
            new_field.fill(0);
#endif

            auto& mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::reference][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_refine.apply_op(prediction<pred_order, true>(new_field, field));
            }

            std::swap(field.array(), new_field.array());
        }

        template <class Mesh, class Fields, std::size_t... Is>
        void update_fields(Mesh& new_mesh, Fields& fields, std::index_sequence<Is...>)
        {
            (update_fields(new_mesh, std::get<Is>(fields)), ...);
        }

        template <class Mesh, class... T>
        void update_fields(Mesh& new_mesh, Field_tuple<T...>& fields)
        {
            update_fields(new_mesh, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
        }

        template <class Mesh, class Field, class... Fields>
        void update_fields(Mesh& new_mesh, Field& field, Fields&... fields)
        {
            update_fields(new_mesh, field);
            update_fields(new_mesh, fields...);
        }

        template <class Mesh>
        void update_fields(Mesh&)
        {
        }
    }

    template <class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        static constexpr std::size_t dim = Tag::dim;
        using mesh_t                     = typename Tag::mesh_t;
        using size_type                  = typename Tag::size_type;
        using mesh_id_t                  = typename Tag::mesh_t::mesh_id_t;
        using cl_type                    = typename Tag::mesh_t::cl_type;

        auto& mesh = tag.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              auto itag = static_cast<size_type>(interval.start + interval.index);
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  if (tag[itag] & static_cast<int>(CellFlag::refine))
                                  {
                                      if (level < mesh.max_level())
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](const auto& stencil)
                                              {
                                                  auto new_index = 2 * index + stencil;
                                                  cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::keep))
                                  {
                                      cl[level][index].add_point(i);
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::coarsen))
                                  {
                                      if (level > mesh.min_level())
                                      {
                                          cl[level - 1][index >> 1].add_point(i >> 1);
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  itag++;
                              }
                          });

        mesh_t new_mesh = {cl, mesh};

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif
        {
            return true;
        }

        detail::update_fields(new_mesh, fields...);
        tag.mesh().swap(new_mesh);
        return false;
    }

    template <class Tag, class Field, class... Fields>
    bool update_field_mr(const Tag& tag, Field& field, Fields&... other_fields)
    {
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        using ca_type   = typename Field::mesh_t::ca_type;

        const auto& mesh = tag.mesh();

        const auto& min_indices = mesh.domain().min_indices();
        const auto& max_indices = mesh.domain().max_indices();

        std::array<int, mesh_t::dim> nb_cells_finest_level;

        for (size_t d = 0; d != max_indices.size(); ++d)
        {
            nb_cells_finest_level[d] = max_indices[d] - min_indices[d];
        }
        ca_type new_ca = update_cell_array_from_tag(mesh[mesh_id_t::cells], tag);
        make_graduation(new_ca, mesh.mpi_neighbourhood(), mesh.periodicity(), nb_cells_finest_level, mesh_t::config::graduation_width);

        mesh_t new_mesh{new_ca, mesh};
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif // SAMURAI_WITH_MPI
        {
            return true;
        }
        detail::update_fields(new_mesh, field, other_fields...);
        field.mesh().swap(new_mesh);
        return false;
    }
}
