// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iterator>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        iterate_over_periodic_ghosts(level,
                                     field,
                                     [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                                     {
                                         field(level, i_ghosts, index_ghosts) = field(level, i_cells, index_cells);
                                     });
    }

    template <class Field, class Func>
    void iterate_over_periodic_ghosts(std::size_t level, Field& field, Func&& copy_values)
    {
#ifdef SAMURAI_WITH_MPI
        using field_value_t = typename Field::value_type;
#endif
        using mesh_id_t        = typename Field::mesh_t::mesh_id_t;
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
            min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
            max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
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

                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        copy_values(i, index, i - shift_interval, index - shift_index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        copy_values(i, index, i + shift_interval, index + shift_index);
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
                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
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
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
#endif
        using mesh_id_t           = typename Tag::mesh_t::mesh_id_t;
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
            min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
            max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
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

                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();

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
                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
            }
        }
    }
}
