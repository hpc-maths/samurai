// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// update_ghost_mr: multiresolution ghost update.
//
// Public API (namespace samurai):
//   update_ghost_mr(field, others...)      time and run the MR ghost update
//   update_ghost_mr(tuple / Field_tuple)   tuple dispatch
//   update_ghost_mr_if_needed(...)         skip fields whose ghosts are current
//
// Implementation (namespace samurai::detail):
//   update_ghost_mr_aggregated(...)        the algorithm below
//   exchange_subdomains_merged(...)        one non-blocking, field-merged
//                                          subdomain exchange per neighbour
//   pack_subdomain_send / unpack_subdomain_recv   (de)serialise one field
//
// This header is included last by update.hpp: the algorithm calls
// outer_subdomain_corner<...> (explicit template argument, so the template must
// be visible at definition), update_ghost_periodic and update_outer_ghosts. It
// pulls only those helper headers, so there is no mutual include with update.hpp.
//
// Algorithm. Per level (top-down then bottom-up), the subdomain ghosts of all
// fields are exchanged in a single non-blocking round per neighbour: fields are
// independent, so they are packed into one buffer per neighbour and the
// receives are posted before the sends. Periodic ghosts keep their per-dimension
// ordering through update_ghost_periodic. What is deliberately NOT merged:
//   - across levels: the projection reference[L] -> proj_cells[L-1] reads ghosts
//     synchronised at level L (inter-level wavefront);
//   - subdomain vs periodic, and periodic dim vs dim: a periodic send reads
//     field(level, i - shift), which is not restricted to mesh.subdomain() and
//     may thus read a subdomain ghost just synchronised, and periodic dimensions
//     accumulate at corners — hence the subdomain -> periodic -> dim ordering.

#include <algorithm>
#include <iterator>
#include <vector>

#include "../algorithm.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../timers.hpp"

#include "update_outer_ghost.hpp" // update_outer_ghosts
#include "update_periodic.hpp"    // update_ghost_periodic
#include "update_subdomain.hpp"   // outer_subdomain_corner

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai::detail
{
#ifdef SAMURAI_WITH_MPI
    // Append, for a single field, the subdomain "send" data for one
    // neighbour into `buf`: the inner-interface cells owned by this rank and
    // then the outer-subdomain corners owned by this rank (the send side of
    // the subdomain ghost exchange).
    template <class Field>
    void pack_subdomain_send(std::size_t level,
                             Field& field,
                             const typename Field::mesh_t::mpi_subdomain_t& neighbour,
                             std::vector<typename Field::value_type>& buf)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh();

        auto out_interface = intersection(mesh[mesh_id_t::reference][level], neighbour.mesh[mesh_id_t::reference][level], mesh.subdomain())
                                 .on(level);
        out_interface(
            [&](const auto& i, const auto& index)
            {
                std::copy(field(level, i, index).begin(), field(level, i, index).end(), std::back_inserter(buf));
            });

        auto subdomain_corners = outer_subdomain_corner<true>(level, field, neighbour);
        for_each_interval(subdomain_corners,
                          [&](const auto, const auto& i, const auto& index)
                          {
                              std::copy(field(level, i, index).begin(), field(level, i, index).end(), std::back_inserter(buf));
                          });
    }

    // Read back, for a single field, the subdomain "recv" data for one
    // neighbour from the iterator `it` (advanced in place). The recv side of
    // the subdomain ghost exchange; it uses the very same intersections as the
    // send side, so the packing order matches by construction.
    template <class Field, class It>
    void unpack_subdomain_recv(std::size_t level, Field& field, const typename Field::mesh_t::mpi_subdomain_t& neighbour, It& it)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh();

        auto in_interface = intersection(neighbour.mesh[mesh_id_t::reference][level],
                                         mesh[mesh_id_t::reference][level],
                                         neighbour.mesh.subdomain())
                                .on(level);
        in_interface(
            [&](const auto& i, const auto& index)
            {
                const auto n = static_cast<std::ptrdiff_t>(i.size() * Field::n_comp);
                std::copy(it, it + n, field(level, i, index).begin());
                it += n;
            });

        auto subdomain_corners = outer_subdomain_corner<false>(level, field, neighbour);
        for_each_interval(subdomain_corners,
                          [&](const auto, const auto& i, const auto& index)
                          {
                              const auto n = static_cast<std::ptrdiff_t>(i.size() * Field::n_comp);
                              std::copy(it, it + n, field(level, i, index).begin());
                              it += n;
                          });
    }
#endif // SAMURAI_WITH_MPI

    // Field-merged, non-blocking subdomain ghost exchange for
    // (level, field, other_fields...). One message per neighbour carries the
    // subdomain ghosts of every field, in pack order [field, other_fields...].
    // Result is identical to exchanging each field on its own.
    template <class Field, class... Fields>
    void
    exchange_subdomains_merged([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& field, [[maybe_unused]] Fields&... other_fields)
    {
#ifdef SAMURAI_WITH_MPI
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        static_assert((std::is_same_v<value_t, typename Fields::value_type> && ...),
                      "aggregated ghost update requires all fields to share the same value_type");

        auto& mesh = field.mesh();
        mpi::communicator world;
        const auto& neighbourhood = mesh.mpi_neighbourhood();
        const std::size_t n       = neighbourhood.size();

        // Symmetric guard: both ranks of a pair evaluate the identical
        // predicate, so they agree on whether a message is exchanged (no
        // deadlock, no mismatch).
        auto active = [&](const auto& neighbour)
        {
            return !mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty();
        };

        // Sized up front so element addresses stay stable for the whole
        // exchange (isend/irecv keep references to these buffers).
        std::vector<std::vector<value_t>> to_send(n);
        std::vector<std::vector<value_t>> to_recv(n);
        std::vector<mpi::request> req;
        req.reserve(2 * n);

        for (std::size_t k = 0; k < n; ++k)
        {
            if (active(neighbourhood[k]))
            {
                req.push_back(world.irecv(neighbourhood[k].rank, world.rank(), to_recv[k]));
            }
        }

        for (std::size_t k = 0; k < n; ++k)
        {
            if (!active(neighbourhood[k]))
            {
                continue;
            }
            pack_subdomain_send(level, field, neighbourhood[k], to_send[k]);
            (pack_subdomain_send(level, other_fields, neighbourhood[k], to_send[k]), ...);
            req.push_back(world.isend(neighbourhood[k].rank, neighbourhood[k].rank, to_send[k]));
        }

        mpi::wait_all(req.begin(), req.end());

        for (std::size_t k = 0; k < n; ++k)
        {
            if (!active(neighbourhood[k]))
            {
                continue;
            }
            auto it = to_recv[k].cbegin();
            unpack_subdomain_recv(level, field, neighbourhood[k], it);
            (unpack_subdomain_recv(level, other_fields, neighbourhood[k], it), ...);
        }
#endif // SAMURAI_WITH_MPI
    }

    // Aggregated counterpart of update_ghost_mr. Same multiresolution
    // top-down / bottom-up structure; only the subdomain exchange is
    // field-merged and non-blocking. The periodic exchange keeps its
    // historic per-dimension ordering (see header note) by delegating to
    // update_ghost_periodic.
    template <class Field, class... Fields>
    void update_ghost_mr_aggregated(Field& field, Fields&... other_fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;

        auto& mesh            = field.mesh();
        auto max_level        = mesh.max_level();
        std::size_t min_level = 0;

        for (std::size_t level = max_level + 1; level-- > min_level;)
        {
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);
            update_outer_ghosts(level, field, other_fields...);
            // Second subdomain sync, required for decomposition independence:
            // update_outer_ghosts recomputes the outer/B.C. ghosts locally, but
            // for non-stripe partitions the owner-computed outer values (see
            // outer_subdomain_corner) must still be redistributed to every rank
            // that references them.
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);

            if (level > min_level)
            {
                auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
                set_at_levelm1.apply_op(variadic_projection(field, other_fields...));
            }
        }

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto expr        = intersection(pred_ghosts, mesh.subdomain(), mesh[mesh_id_t::all_cells][level - 1]).on(level);

            expr.apply_op(variadic_prediction<pred_order, false>(field, other_fields...));
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);
        }

        field.ghosts_updated() = true;
        ((other_fields.ghosts_updated() = true), ...);
    }
} // namespace samurai::detail

namespace samurai
{
    template <class Field>
    void update_ghost_mr_if_needed(Field& field)
    {
        if (!field.ghosts_updated())
        {
            update_ghost_mr(field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_mr_if_needed(Field& field, Fields&... other_fields)
    {
        update_ghost_mr_if_needed(field);
        update_ghost_mr_if_needed(other_fields...);
    }

    // Public entry point: times the update and delegates to the aggregated
    // implementation above.
    template <class Field, class... Fields>
    void update_ghost_mr(Field& field, Fields&... other_fields)
    {
        ScopedTimer timer_ghosts("ghost update");
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh();

        detail::update_ghost_mr_aggregated(field, other_fields...);

        timer_ghosts.set_cells(mesh.nb_cells(mesh_id_t::cells));
    }

    SAMURAI_INLINE void update_ghost_mr()
    {
    }

    template <class... T>
    SAMURAI_INLINE void update_ghost_mr(std::tuple<T...>& fields)
    {
        std::apply(
            [](T&... tupleArgs)
            {
                update_ghost_mr(tupleArgs...);
            },
            fields);
    }

    template <class... T>
    SAMURAI_INLINE void update_ghost_mr(Field_tuple<T...>& fields)
    {
        update_ghost_mr(fields.elements());
    }
}
