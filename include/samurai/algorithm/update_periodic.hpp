// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Periodic ghost and tag updates.
//
// Along a periodic dimension d, the ghosts beyond the max face of the domain
// take the values of the cells inside the min face, shifted by one period, and
// the ghosts below the min face those of the cells inside the max face. The
// dimensions are processed one after the other: the bands of a dimension span
// the ghost width beyond the domain in the other dimensions, so a corner ghost
// is filled in the last periodic dimension from a ghost filled in an earlier
// one. Within a dimension, a rank first copies its own cells into its own
// ghosts, then exchanges with its MPI neighbourhood the copies that cross a
// subdomain boundary, in one message per neighbour for all the fields, and no
// message at all when there is nothing to copy (see neighbour_exchange.hpp for
// why both ranks of a pair skip the same messages).

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"
#include "neighbour_exchange.hpp"

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai::detail
{
    // Where the periodic copies of one level can happen for this rank: the
    // faces of the domain and the bounding box of the rank's reference cells.
    template <class Mesh>
    struct periodic_frame
    {
        static constexpr std::size_t dim = Mesh::dim;
        using value_t                    = typename Mesh::interval_t::value_t;
        using coords_t                   = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        // `level` must hold reference cells on this rank.
        periodic_frame(const Mesh& mesh, std::size_t level)
            : level(level)
            , ghost_width(mesh.ghost_width())
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            const auto& domain        = mesh.domain();
            const std::size_t delta_l = domain.level() - level;
            const auto domain_min     = domain.min_indices();
            const auto domain_max     = domain.max_indices();
            const auto& own           = mesh[mesh_id_t::reference][level];
            const auto own_min        = own.min_indices();
            const auto own_max        = own.max_indices();
            for (std::size_t k = 0; k < dim; ++k)
            {
                min_face[k]  = domain_min[k] >> delta_l;
                max_face[k]  = domain_max[k] >> delta_l;
                cells_min[k] = own_min[k];
                cells_max[k] = own_max[k];
            }
        }

        std::size_t level;
        value_t ghost_width;
        coords_t min_face;
        coords_t max_face;
        coords_t cells_min; // bounding box of the rank's reference cells
        coords_t cells_max;
    };

    // The four bands of the periodic copies along dimension d at one level, one
    // ghost width thick along d, and the period of the domain along d.
    //
    // Along the other dimensions the bands span the domain plus the ghost width,
    // clipped to the bounding box of this rank's reference cells. The clipping
    // removes no copy this rank takes part in: every copy it makes, sends or
    // receives reads or writes one of its own cells, and the periodic shift does
    // not move cells along the other dimensions. It keeps the cost of the bands
    // proportional to the subdomain instead of the whole domain.
    template <class Mesh>
    struct periodic_bands
    {
        static constexpr std::size_t dim = Mesh::dim;
        using lca_type                   = typename Mesh::lca_type;
        using value_t                    = typename Mesh::interval_t::value_t;
        using coords_t                   = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        periodic_bands(const periodic_frame<Mesh>& frame, std::size_t d)
            : min_m(frame.level)
            , max_m(frame.level)
            , min_p(frame.level)
            , max_p(frame.level)
        {
            const value_t gw = frame.ghost_width;
            const value_t lo = frame.min_face[d];
            const value_t hi = frame.max_face[d];

            shift.fill(0);
            shift[d] = hi - lo;

            coords_t min_corner;
            coords_t max_corner;
            bool overlap = true;
            for (std::size_t k = 0; k < dim; ++k)
            {
                min_corner[k] = std::max(frame.min_face[k] - gw, frame.cells_min[k]);
                max_corner[k] = std::min(frame.max_face[k] + gw, frame.cells_max[k]);
                overlap       = overlap && (k == d || min_corner[k] < max_corner[k]);
            }
            // Every copy this rank takes part in reads or writes one of its
            // cells lying within the ghost width of a face along d.
            const bool near_min = frame.cells_min[d] < lo + gw && lo - gw < frame.cells_max[d];
            const bool near_max = frame.cells_min[d] < hi + gw && hi - gw < frame.cells_max[d];
            active              = overlap && (near_min || near_max);
            if (!active)
            {
                return;
            }

            auto band = [&](value_t start, value_t end)
            {
                min_corner[d] = start;
                max_corner[d] = end;
                return lca_type(frame.level, Box<value_t, dim>(min_corner, max_corner));
            };
            min_m = band(lo - gw, lo);
            max_m = band(hi - gw, hi);
            min_p = band(lo, lo + gw);
            max_p = band(hi, hi + gw);
        }

        // false when this rank takes part in no periodic copy along d at this
        // level. Every copy involving the rank would then be empty, also as
        // evaluated by its neighbours, so all of them skip the dimension.
        bool active;
        lca_type min_m; // outside the domain, below its min face
        lca_type max_m; // inside the domain, below its max face
        lca_type min_p; // inside the domain, above its min face
        lca_type max_p; // outside the domain, above its max face
        coords_t shift;
    };

    // Traverse the periodic copies along the dimension of `bands` from the cells
    // of `src` to the cells of `dst` (two level cell arrays of the same level):
    // f(i, index, i_src, index_src) for each interval (i, index) of dst whose
    // value is the one of (i_src, index_src) in src.
    template <class Mesh, class LCA_src, class LCA_dst, class Func>
    void for_each_periodic_copy(const periodic_bands<Mesh>& bands, const LCA_src& src, const LCA_dst& dst, Func&& f)
    {
        const auto shift_interval = bands.shift[0];
        const auto shift_index    = xt::view(bands.shift, xt::range(1, _));

        auto beyond_max = intersection(translate(intersection(src, bands.min_p), bands.shift), intersection(dst, bands.max_p));
        beyond_max(
            [&](const auto& i, const auto& index)
            {
                f(i, index, i - shift_interval, index - shift_index);
            });
        auto below_min = intersection(translate(intersection(src, bands.max_m), -bands.shift), intersection(dst, bands.min_m));
        below_min(
            [&](const auto& i, const auto& index)
            {
                f(i, index, i + shift_interval, index + shift_index);
            });
    }

    // The periodic copies from one cell collection to another, evaluated once:
    // dst.interval(k) takes the values of src.interval(k).
    template <class Mesh>
    struct periodic_copies
    {
        exchange_intervals<Mesh> dst;
        exchange_intervals<Mesh> src;
    };

    template <class Mesh, class LCA_src, class LCA_dst>
    periodic_copies<Mesh> make_periodic_copies(const periodic_bands<Mesh>& bands, const LCA_src& src, const LCA_dst& dst)
    {
        periodic_copies<Mesh> copies;
        for_each_periodic_copy(bands,
                               src,
                               dst,
                               [&](const auto& i, const auto& index, const auto& i_src, const auto& index_src)
                               {
                                   copies.dst.push_back(i, index);
                                   copies.src.push_back(i_src, index_src);
                               });
        return copies;
    }

#ifdef SAMURAI_WITH_MPI
    // The periodic copies crossing the boundary between this subdomain and each
    // of its MPI neighbours: send[k] goes from this rank's cells to the ghosts
    // of neighbour k, recv[k] from the cells of neighbour k to this rank's
    // ghosts. send[k] on this rank and recv[.] on neighbour k are the same set,
    // evaluated from the two copies of the two meshes.
    template <class Mesh>
    struct periodic_neighbour_copies
    {
        std::vector<periodic_copies<Mesh>> send;
        std::vector<periodic_copies<Mesh>> recv;
        std::vector<std::size_t> send_cells;
        std::vector<std::size_t> recv_cells;

        periodic_neighbour_copies(const Mesh& mesh, std::size_t level, const periodic_bands<Mesh>& bands)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            const auto& own = mesh[mesh_id_t::reference][level];
            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                const auto& other = neighbour.mesh[mesh_id_t::reference][level];
                send.push_back(make_periodic_copies(bands, own, other));
                recv.push_back(make_periodic_copies(bands, other, own));
                send_cells.push_back(send.back().src.n_cells());
                recv_cells.push_back(recv.back().dst.n_cells());
            }
        }
    };

    inline std::vector<std::size_t> scale_counts(const std::vector<std::size_t>& cells, std::size_t values_per_cell)
    {
        std::vector<std::size_t> counts(cells.size());
        for (std::size_t k = 0; k < cells.size(); ++k)
        {
            counts[k] = cells[k] * values_per_cell;
        }
        return counts;
    }
#endif // SAMURAI_WITH_MPI

    // Exchange with the MPI neighbourhood the periodic ghosts along the
    // dimension of `bands` of every field, one message per neighbour.
    template <class Field, class... Fields>
    void exchange_periodic_ghosts([[maybe_unused]] std::size_t level,
                                  [[maybe_unused]] const periodic_bands<typename Field::mesh_t>& bands,
                                  [[maybe_unused]] Field& field,
                                  [[maybe_unused]] Fields&... other_fields)
    {
#ifdef SAMURAI_WITH_MPI
        using value_t = typename Field::value_type;
        static_assert((std::is_same_v<value_t, typename Fields::value_type> && ...),
                      "a merged periodic exchange requires all fields to share the same value_type");

        const auto& mesh = field.mesh();
        if (mesh.mpi_neighbourhood().empty())
        {
            return;
        }

        const periodic_neighbour_copies copies(mesh, level, bands);
        constexpr std::size_t values_per_cell = (Field::n_comp + ... + Fields::n_comp);

        neighbour_exchange<value_t>(
            mesh.mpi_neighbourhood(),
            scale_counts(copies.send_cells, values_per_cell),
            scale_counts(copies.recv_cells, values_per_cell),
            [&](std::size_t k, auto& buf)
            {
                pack_intervals(level, field, copies.send[k].src, buf);
                (pack_intervals(level, other_fields, copies.send[k].src, buf), ...);
            },
            [&](std::size_t k, auto& it)
            {
                unpack_intervals(level, field, copies.recv[k].dst, it);
                (unpack_intervals(level, other_fields, copies.recv[k].dst, it), ...);
            });
#endif // SAMURAI_WITH_MPI
    }

    // Periodic ghost update at one level of fields sharing a value_type.
    template <class Field, class... Fields>
    void update_ghost_periodic_merged(std::size_t level, Field& field, Fields&... other_fields)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        const auto& mesh = field.mesh();
        // An empty level has nothing to copy, to send or to receive, and every
        // neighbour reaches the same conclusion from its copy of this mesh.
        if (!mesh.is_periodic() || mesh[mesh_id_t::reference][level].empty())
        {
            return;
        }

        const auto& own = mesh[mesh_id_t::reference][level];
        const periodic_frame frame(mesh, level);
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                const periodic_bands bands(frame, d);
                if (!bands.active)
                {
                    continue;
                }
                const auto local = make_periodic_copies(bands, own, own);
                auto copy        = [&](auto& f)
                {
                    for (std::size_t k = 0; k < local.dst.size(); ++k)
                    {
                        f(level, local.dst.interval(k), local.dst.index(k)) = f(level, local.src.interval(k), local.src.index(k));
                    }
                };
                copy(field);
                (copy(other_fields), ...);

                exchange_periodic_ghosts(level, bands, field, other_fields...);
            }
        }
    }
} // namespace samurai::detail

namespace samurai
{
    // Apply copy_values(i_ghosts, index_ghosts, i_cells, index_cells) to the
    // periodic copies inside this subdomain, and exchange with the MPI
    // neighbourhood the periodic ghosts of `field` that cross a subdomain
    // boundary.
    template <class Field, class Func>
    void iterate_over_periodic_ghosts(std::size_t level, Field& field, Func&& copy_values)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        const auto& mesh = field.mesh();
        if (!mesh.is_periodic() || mesh[mesh_id_t::reference][level].empty())
        {
            return;
        }

        const auto& own = mesh[mesh_id_t::reference][level];
        const detail::periodic_frame frame(mesh, level);
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                const detail::periodic_bands bands(frame, d);
                if (!bands.active)
                {
                    continue;
                }
                detail::for_each_periodic_copy(bands, own, own, copy_values);
                detail::exchange_periodic_ghosts(level, bands, field);
            }
        }
    }

    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        detail::update_ghost_periodic_merged(level, field);
    }

    // Fields sharing a value_type are exchanged together: one message per
    // neighbour and per periodic dimension for all of them. The fields are
    // independent, so the result is the one of updating them one by one.
    template <class Field, class... Fields>
    void update_ghost_periodic(std::size_t level, Field& field, Fields&... other_fields)
    {
        if constexpr ((std::is_same_v<typename Field::value_type, typename Fields::value_type> && ...))
        {
            detail::update_ghost_periodic_merged(level, field, other_fields...);
        }
        else
        {
            update_ghost_periodic(level, field);
            update_ghost_periodic(level, other_fields...);
        }
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

    // Periodic tag update at one level: a cell and its periodic images end with
    // the bitwise or of their tags.
    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
        using mesh_id_t = typename Tag::mesh_t::mesh_id_t;

        const auto& mesh = tag.mesh();
        if (!mesh.is_periodic() || mesh[mesh_id_t::reference][level].empty())
        {
            return;
        }

        const auto& own = mesh[mesh_id_t::reference][level];
        const detail::periodic_frame frame(mesh, level);
        for (std::size_t d = 0; d < Tag::dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                const detail::periodic_bands bands(frame, d);
                if (!bands.active)
                {
                    continue;
                }
                detail::for_each_periodic_copy(bands,
                                               own,
                                               own,
                                               [&](const auto& i, const auto& index, const auto& i_src, const auto& index_src)
                                               {
                                                   tag(level, i, index) |= tag(level, i_src, index_src);
                                                   tag(level, i_src, index_src) |= tag(level, i, index);
                                               });
#ifdef SAMURAI_WITH_MPI
                if (mesh.mpi_neighbourhood().empty())
                {
                    continue;
                }
                using value_t = typename Tag::value_type;

                const detail::periodic_neighbour_copies copies(mesh, level, bands);
                const auto send_count = detail::scale_counts(copies.send_cells, Tag::n_comp);
                const auto recv_count = detail::scale_counts(copies.recv_cells, Tag::n_comp);

                auto or_unpack = [&](const auto& intervals, auto& it)
                {
                    for (std::size_t k = 0; k < intervals.size(); ++k)
                    {
                        for (auto& t : tag(level, intervals.interval(k), intervals.index(k)))
                        {
                            t |= *it;
                            ++it;
                        }
                    }
                };

                // First pass: the ghosts of a neighbour gather the tags of the
                // cells they are the periodic image of.
                detail::neighbour_exchange<value_t>(
                    mesh.mpi_neighbourhood(),
                    send_count,
                    recv_count,
                    [&](std::size_t k, auto& buf)
                    {
                        detail::pack_intervals(level, tag, copies.send[k].src, buf);
                    },
                    [&](std::size_t k, auto& it)
                    {
                        or_unpack(copies.recv[k].dst, it);
                    });
                // Second pass, along the same copies in the other direction: the
                // cells gather back the tags their periodic images now hold.
                detail::neighbour_exchange<value_t>(
                    mesh.mpi_neighbourhood(),
                    recv_count,
                    send_count,
                    [&](std::size_t k, auto& buf)
                    {
                        detail::pack_intervals(level, tag, copies.recv[k].dst, buf);
                    },
                    [&](std::size_t k, auto& it)
                    {
                        or_unpack(copies.send[k].src, it);
                    });
#endif // SAMURAI_WITH_MPI
            }
        }
    }
} // namespace samurai
