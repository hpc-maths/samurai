// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Building blocks shared by the ghost and tag exchanges with the MPI
// neighbourhood (update_periodic.hpp, update_subdomain.hpp,
// update_ghost_mr.hpp):
//
//   exchange_intervals   the intervals one exchange reads or writes for one
//                        neighbour, in traversal order, evaluated once and then
//                        reused for every field of the exchange;
//   neighbour_exchange   one round of point-to-point messages with the
//                        neighbourhood, which skips the empty ones.
//
// Skipping a message without telling the receiver is only correct when the
// receiver reaches the same decision on its own. Every exchange that goes
// through neighbour_exchange guarantees it the same way: the cells rank A sends
// to rank B and the cells B receives from A are ONE set expression, evaluated by
// A on (its own mesh, its copy of B's mesh) and by B on (its copy of A's mesh,
// its own mesh). The neighbour copies are exact (Mesh_base::update_mesh_neighbour
// serialises every cell array, the domain, the subdomain and the
// configuration), so both ranks obtain the same cells, hence the same message
// size, and they agree on which messages are empty without communicating. The
// unpacking already relied on this: it reads the received buffer through the
// receiver's own evaluation of the set.

#include <cassert>
#include <cstddef>
#include <vector>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

namespace samurai::detail
{
    // Intervals of one level, in the order a set expression yielded them.
    template <class Mesh>
    class exchange_intervals
    {
      public:

        using interval_t = typename Mesh::interval_t;
        using coord_t    = typename Mesh::lca_type::coord_type;

        template <class Index>
        void push_back(const interval_t& i, const Index& index)
        {
            m_intervals.push_back(i);
            m_indices.emplace_back(index);
            m_n_cells += i.size();
        }

        std::size_t size() const
        {
            return m_intervals.size();
        }

        bool empty() const
        {
            return m_intervals.empty();
        }

        // Number of cells covered by the intervals.
        std::size_t n_cells() const
        {
            return m_n_cells;
        }

        const interval_t& interval(std::size_t k) const
        {
            return m_intervals[k];
        }

        const coord_t& index(std::size_t k) const
        {
            return m_indices[k];
        }

      private:

        std::vector<interval_t> m_intervals;
        std::vector<coord_t> m_indices;
        std::size_t m_n_cells = 0;
    };

    // Append the values of `field` on `intervals` to `buf`.
    template <class Field>
    void pack_intervals(std::size_t level,
                        Field& field,
                        const exchange_intervals<typename Field::mesh_t>& intervals,
                        std::vector<typename Field::value_type>& buf)
    {
        for (std::size_t k = 0; k < intervals.size(); ++k)
        {
            const auto values = field(level, intervals.interval(k), intervals.index(k));
            buf.insert(buf.end(), values.begin(), values.end());
        }
    }

    // Read the values of `field` on `intervals` from `it` (advanced in place).
    template <class Field, class It>
    void unpack_intervals(std::size_t level, Field& field, const exchange_intervals<typename Field::mesh_t>& intervals, It& it)
    {
        for (std::size_t k = 0; k < intervals.size(); ++k)
        {
            auto values = field(level, intervals.interval(k), intervals.index(k));
            for (auto& v : values)
            {
                v = *it;
                ++it;
            }
        }
    }

#ifdef SAMURAI_WITH_MPI
    // One round of messages with the neighbourhood: neighbour k is sent
    // send_count[k] values and is received recv_count[k] values. A message whose
    // count is zero is neither sent nor received; the caller guarantees that the
    // two ranks of a pair compute the same count for the message between them
    // (see the header note).
    //
    // pack(k, buf) appends the send_count[k] values for neighbour k to buf;
    // unpack(k, it) consumes the recv_count[k] values received from neighbour k.
    // Every receive is posted before any send, and the messages are unpacked in
    // neighbourhood order once all of them have arrived, so a cell written by
    // several neighbours ends with the same value as with blocking receives in
    // neighbourhood order.
    template <class value_t, class Neighbourhood, class Pack, class Unpack>
    void neighbour_exchange(const Neighbourhood& neighbourhood,
                            const std::vector<std::size_t>& send_count,
                            const std::vector<std::size_t>& recv_count,
                            Pack&& pack,
                            Unpack&& unpack)
    {
        namespace mpi = boost::mpi;
        mpi::communicator world;

        const std::size_t n = neighbourhood.size();
        // Sized up front: the requests keep pointers to these buffers.
        std::vector<std::vector<value_t>> to_send(n);
        std::vector<std::vector<value_t>> to_recv(n);
        std::vector<mpi::request> req;
        req.reserve(2 * n);

        for (std::size_t k = 0; k < n; ++k)
        {
            if (recv_count[k] > 0)
            {
                to_recv[k].resize(recv_count[k]);
                req.push_back(world.irecv(neighbourhood[k].rank, world.rank(), to_recv[k].data(), static_cast<int>(recv_count[k])));
            }
        }

        for (std::size_t k = 0; k < n; ++k)
        {
            if (send_count[k] > 0)
            {
                to_send[k].reserve(send_count[k]);
                pack(k, to_send[k]);
                assert(to_send[k].size() == send_count[k]);
                req.push_back(world.isend(neighbourhood[k].rank, neighbourhood[k].rank, to_send[k].data(), static_cast<int>(send_count[k])));
            }
        }

        mpi::wait_all(req.begin(), req.end());

        for (std::size_t k = 0; k < n; ++k)
        {
            if (recv_count[k] > 0)
            {
                auto it = to_recv[k].cbegin();
                unpack(k, it);
                assert(it == to_recv[k].cend());
            }
        }
    }
#endif // SAMURAI_WITH_MPI
} // namespace samurai::detail
