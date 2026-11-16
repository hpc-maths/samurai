// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"
#include "../entry_table.hpp"

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace detail
    {
        // The cells of `level` within the ghost width of the lower (min) and
        // upper (max) faces of the domain in the direction d, below the face
        // (_m) and above it (_p), and the translation from one face to the other.
        template <class Mesh>
        struct PeriodicBands
        {
            using lca_type = typename Mesh::lca_type;
            using value_t  = typename Mesh::interval_t::value_t;

            lca_type min_m;
            lca_type max_m;
            lca_type min_p;
            lca_type max_p;
            xt::xtensor_fixed<value_t, xt::xshape<Mesh::dim>> shift;

            PeriodicBands(const Mesh& mesh, std::size_t level, std::size_t d)
            {
                using box_t               = Box<value_t, Mesh::dim>;
                constexpr std::size_t dim = Mesh::dim;

                const auto& domain        = mesh.domain();
                const auto& min_indices   = domain.min_indices();
                const auto& max_indices   = domain.max_indices();
                const std::size_t delta_l = domain.level() - level;
                const auto ghost_width    = mesh.ghost_width();

                xt::xtensor_fixed<value_t, xt::xshape<dim>> min_corner;
                xt::xtensor_fixed<value_t, xt::xshape<dim>> max_corner;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    min_corner[k] = (min_indices[k] >> delta_l) - ghost_width;
                    max_corner[k] = (max_indices[k] >> delta_l) + ghost_width;
                    shift[k]      = 0;
                }
                shift[d] = (max_indices[d] - min_indices[d]) >> delta_l;

                auto band = [&](value_t start)
                {
                    min_corner[d] = start;
                    max_corner[d] = start + ghost_width;
                    return lca_type(level, box_t(min_corner, max_corner));
                };
                min_m = band((min_indices[d] >> delta_l) - ghost_width);
                max_m = band((max_indices[d] >> delta_l) - ghost_width);
                min_p = band(min_indices[d] >> delta_l);
                max_p = band(max_indices[d] >> delta_l);
            }

            // Calls f(i, index, i_image, index_image) for each interval of the
            // cells of `ghosts` outside the domain along d whose periodic image
            // is in `images`.
            template <class Ghosts, class Images, class Func>
            void for_each_pair(const Ghosts& ghosts, const Images& images, Func&& f) const
            {
                const auto level       = min_m.level();
                const auto shift_x     = shift[0];
                const auto shift_index = xt::view(shift, xt::range(1, _));

                auto above = intersection(translate(intersection(images[level], min_p), shift), intersection(ghosts[level], max_p));
                above(
                    [&](const auto& i, const auto& index)
                    {
                        f(i, index, i - shift_x, index - shift_index);
                    });
                auto below = intersection(translate(intersection(images[level], max_m), -shift), intersection(ghosts[level], min_m));
                below(
                    [&](const auto& i, const auto& index)
                    {
                        f(i, index, i + shift_x, index + shift_index);
                    });
            }
        };

        // The periodic ghosts of `level` in the direction d whose image this
        // rank holds: at[0] is the row of ghosts, at[1] the row of their images.
        struct periodic_copies
        {
            using entry_type = Rows<2>;

            std::size_t level;
            std::size_t d;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                const auto& ref = mesh[Mesh::mesh_id_t::reference];
                const PeriodicBands<Mesh> bands(mesh, level, d);
                bands.for_each_pair(
                    ref,
                    ref,
                    [&](const auto& i, const auto& index, const auto& i_image, const auto& index_image)
                    {
                        emit(entry_type{
                            i.size(),
                            {memory_offset(mesh, {level, i.start, index}), memory_offset(mesh, {level, i_image.start, index_image})}
                        });
                    });
            }
        };

        // The cells of `level` whose periodic image in the direction d is a
        // ghost of the neighbour-th neighbour, in the order it receives them.
        struct periodic_sent_cells
        {
            using entry_type = Rows<1>;

            std::size_t level;
            std::size_t d;
            std::size_t neighbour;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;
                const PeriodicBands<Mesh> bands(mesh, level, d);
                bands.for_each_pair(mesh.mpi_neighbourhood()[neighbour].mesh[mesh_id_t::reference],
                                    mesh[mesh_id_t::reference],
                                    [&](const auto& i, const auto&, const auto& i_image, const auto& index_image)
                                    {
                                        emit(entry_type{i.size(), {memory_offset(mesh, {level, i_image.start, index_image})}});
                                    });
            }
        };

        // The periodic ghosts of `level` in the direction d whose image belongs
        // to the neighbour-th neighbour, in the order it sends them.
        struct periodic_received_ghosts
        {
            using entry_type = Rows<1>;

            std::size_t level;
            std::size_t d;
            std::size_t neighbour;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;
                const PeriodicBands<Mesh> bands(mesh, level, d);
                bands.for_each_pair(mesh[mesh_id_t::reference],
                                    mesh.mpi_neighbourhood()[neighbour].mesh[mesh_id_t::reference],
                                    [&](const auto& i, const auto& index, const auto&, const auto&)
                                    {
                                        emit(entry_type{i.size(), {memory_offset(mesh, {level, i.start, index})}});
                                    });
            }
        };
    }

    // Copies the periodic ghosts of `level` from their image, one periodic
    // direction after the other, and exchanges those whose image is held by a
    // neighbour. The components of a cell are next to each other.
    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        constexpr std::size_t nc = Field::n_comp;
        const auto& mesh         = field.mesh();
        auto* data               = field.data();
#ifdef SAMURAI_WITH_MPI
        using value_t = typename Field::value_type;
        mpi::communicator world;
        const auto& neighbours = mesh.mpi_neighbourhood();
#endif
        auto copy_images = [&](const auto& rows)
        {
            std::copy(data + rows.at[1] * nc, data + (rows.at[1] + rows.len) * nc, data + rows.at[0] * nc);
        };
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (!mesh.is_periodic(d))
            {
                continue;
            }
            for_each_entry(mesh, detail::periodic_copies{level, d}, copy_images);
#ifdef SAMURAI_WITH_MPI
            if (neighbours.empty())
            {
                continue;
            }
            // Scoped to the direction: waiting again on completed boost::mpi
            // serialized requests corrupts the MPI request state.
            std::vector<std::vector<value_t>> out(neighbours.size());
            std::vector<mpi::request> req;
            req.reserve(neighbours.size());
            for (std::size_t k = 0; k < neighbours.size(); ++k)
            {
                for_each_entry(mesh,
                               detail::periodic_sent_cells{level, d, k},
                               [&](const auto& rows)
                               {
                                   out[k].insert(out[k].end(), data + rows.at[0] * nc, data + (rows.at[0] + rows.len) * nc);
                               });
                req.push_back(world.isend(neighbours[k].rank, neighbours[k].rank, out[k]));
            }
            std::vector<value_t> in;
            for (std::size_t k = 0; k < neighbours.size(); ++k)
            {
                world.recv(neighbours[k].rank, world.rank(), in);
                auto it = in.cbegin();
                for_each_entry(mesh,
                               detail::periodic_received_ghosts{level, d, k},
                               [&](const auto& rows)
                               {
                                   const auto n = static_cast<std::ptrdiff_t>(rows.len * nc);
                                   std::copy(it, it + n, data + rows.at[0] * nc);
                                   it += n;
                               });
            }
            mpi::wait_all(req.begin(), req.end());
#endif
        }
    }

    // Calls copy_values(i_ghosts, index_ghosts, i_cells, index_cells) for each
    // interval of periodic ghosts of `level` whose image this rank holds, one
    // periodic direction after the other, and exchanges the values of the
    // others with the neighbours.
    template <class Field, class Func>
    void iterate_over_periodic_ghosts(std::size_t level, Field& field, Func&& copy_values)
    {
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        auto& mesh      = field.mesh();
        const auto& ref = mesh[mesh_id_t::reference];
#ifdef SAMURAI_WITH_MPI
        using field_value_t = typename Field::value_type;
        mpi::communicator world;
        const auto& neighbours = mesh.mpi_neighbourhood();
#endif
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (!mesh.is_periodic(d))
            {
                continue;
            }
            const detail::PeriodicBands<mesh_t> bands(mesh, level, d);
            bands.for_each_pair(ref, ref, copy_values);
#ifdef SAMURAI_WITH_MPI
            // These MUST be scoped to the current dimension: reusing the same
            // request vector across dimensions would call wait_all() again on
            // requests already completed in the previous dimension, which is
            // undefined for boost::mpi's serialized-vector requests and
            // corrupts the MPI request state (intermittent hangs/crashes).
            std::vector<mpi::request> req;
            req.reserve(neighbours.size());
            std::vector<std::vector<field_value_t>> field_data_out(neighbours.size());
            std::vector<field_value_t> field_data_in;

            for (std::size_t k = 0; k < neighbours.size(); ++k)
            {
                // the cells whose image is a ghost of the neighbour
                bands.for_each_pair(neighbours[k].mesh[mesh_id_t::reference],
                                    ref,
                                    [&](const auto&, const auto&, const auto& i_cells, const auto& index_cells)
                                    {
                                        const auto& field_data = field(level, i_cells, index_cells);
                                        std::copy(field_data.begin(), field_data.end(), std::back_inserter(field_data_out[k]));
                                    });
                req.push_back(world.isend(neighbours[k].rank, neighbours[k].rank, field_data_out[k]));
            }
            for (const auto& neighbour : neighbours)
            {
                world.recv(neighbour.rank, world.rank(), field_data_in);
                auto it = field_data_in.cbegin();
                // the ghosts whose image belongs to the neighbour
                bands.for_each_pair(ref,
                                    neighbour.mesh[mesh_id_t::reference],
                                    [&](const auto& i, const auto& index, const auto&, const auto&)
                                    {
                                        auto ghosts  = field(level, i, index);
                                        const auto n = std::ssize(ghosts);
                                        std::copy(it, it + n, ghosts.begin());
                                        it += n;
                                    });
            }
            mpi::wait_all(req.begin(), req.end());
#endif
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

    // Merges the tags of each periodic ghost of `level` and of its image: both
    // get their union. The images held by a neighbour are exchanged in two
    // passes, ghosts from images then images from ghosts.
    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
        using mesh_t    = typename Tag::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        auto& mesh      = tag.mesh();
        const auto& ref = mesh[mesh_id_t::reference];
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
        mpi::communicator world;
        const auto& neighbours = mesh.mpi_neighbourhood();

        // One pass with every neighbour: with ghosts_from_images, the ghosts
        // get the tags of their image, otherwise the images get those of their ghost.
        auto exchange = [&](const detail::PeriodicBands<mesh_t>& bands, bool ghosts_from_images)
        {
            // Scoped to one exchange on purpose: waiting again on completed
            // boost::mpi serialized requests corrupts the MPI request state
            // (intermittent hangs/crashes).
            std::vector<mpi::request> req;
            req.reserve(neighbours.size());
            std::vector<std::vector<tag_value_type>> tag_data_out(neighbours.size());
            std::vector<tag_value_type> tag_data_in;

            for (std::size_t k = 0; k < neighbours.size(); ++k)
            {
                const auto& neighbour_ref = neighbours[k].mesh[mesh_id_t::reference];
                auto send                 = [&](const auto& i, const auto& index)
                {
                    const auto& tag_data = tag(level, i, index);
                    std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[k]));
                };
                if (ghosts_from_images)
                {
                    // my cells whose image is a ghost of the neighbour
                    bands.for_each_pair(neighbour_ref,
                                        ref,
                                        [&](const auto&, const auto&, const auto& i_image, const auto& index_image)
                                        {
                                            send(i_image, index_image);
                                        });
                }
                else
                {
                    // my ghosts whose image belongs to the neighbour
                    bands.for_each_pair(ref,
                                        neighbour_ref,
                                        [&](const auto& i, const auto& index, const auto&, const auto&)
                                        {
                                            send(i, index);
                                        });
                }
                req.push_back(world.isend(neighbours[k].rank, neighbours[k].rank, tag_data_out[k]));
            }
            for (const auto& neighbour : neighbours)
            {
                const auto& neighbour_ref = neighbour.mesh[mesh_id_t::reference];
                world.recv(neighbour.rank, world.rank(), tag_data_in);
                auto it    = tag_data_in.cbegin();
                auto merge = [&](const auto& i, const auto& index)
                {
                    for (tag_value_type& value : tag(level, i, index))
                    {
                        value |= *it;
                        ++it;
                    }
                };
                if (ghosts_from_images)
                {
                    bands.for_each_pair(ref,
                                        neighbour_ref,
                                        [&](const auto& i, const auto& index, const auto&, const auto&)
                                        {
                                            merge(i, index);
                                        });
                }
                else
                {
                    bands.for_each_pair(neighbour_ref,
                                        ref,
                                        [&](const auto&, const auto&, const auto& i_image, const auto& index_image)
                                        {
                                            merge(i_image, index_image);
                                        });
                }
            }
            mpi::wait_all(req.begin(), req.end());
        };
#endif
        for (std::size_t d = 0; d < Tag::dim; ++d)
        {
            if (!mesh.is_periodic(d))
            {
                continue;
            }
            const detail::PeriodicBands<mesh_t> bands(mesh, level, d);
            bands.for_each_pair(ref,
                                ref,
                                [&](const auto& i, const auto& index, const auto& i_image, const auto& index_image)
                                {
                                    tag(level, i, index) |= tag(level, i_image, index_image);
                                    tag(level, i_image, index_image) |= tag(level, i, index);
                                });
#ifdef SAMURAI_WITH_MPI
            exchange(bands, true);
            exchange(bands, false);
#endif
        }
    }
}
