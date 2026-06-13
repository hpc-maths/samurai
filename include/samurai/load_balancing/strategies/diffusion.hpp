// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Diffusion load balancing, nD, by interface layers.
 *
 * Two phases:
 *
 *  1. Flux computation (private). The processes form a graph (the MPI
 *     neighbourhood). We solve a discrete heat equation on that graph: at every
 *     iteration each process exchanges its current load with ITS NEIGHBOURS ONLY
 *     and updates a per-edge flux with the generalized Cybenko coefficient
 *
 *         t_j = (load_j - load_i) / (max(deg_i, deg_j) + 1)
 *
 *     (Cybenko 1989). The 1/(max(deg)+1) factor guarantees stability — the fixed
 *     0.5 coefficient of the previous implementation could oscillate. The only
 *     collective is one boolean all_reduce per iteration to detect convergence
 *     (plus one all_reduce of the total load, once, to set the convergence
 *     scale): there is NO all_gather of the loads. Fluxes below
 *     `flux_threshold * mean_load` are zeroed to avoid micro-migrations.
 *     Convergence towards the global balance is geometric in the spectral gap of
 *     the process graph; since AMR calls the balancer again at every adaptation,
 *     a partial convergence per call is acceptable.
 *
 *  2. Layer assignment (nD). A negative flux fluxes[j] means "I must shed
 *     |fluxes[j]| of load to neighbour j". We give away the cells closest to j
 *     first, then progressively deeper layers, so the ceded region stays
 *     connected to the interface (no islands). The cession direction is the
 *     dominant cardinal axis of (barycentre_i - barycentre_j); a diagonal is
 *     split into its cardinal components, treated one after the other. Layers are
 *     built by set algebra at the coarsest level (`min_level`) and projected onto
 *     every actual level with `.on(level)`, which makes the whole construction
 *     dimension-agnostic and level-jump aware. The frontier between subdomains is
 *     therefore a staircase, not a straight line (the straight-line / row-snapping
 *     constraint was exactly what limited the old version to 2D bands).
 *
 * If the interface is exhausted before the requested flux is met, the deficit is
 * accumulated in `last_unmet_flux()` (surfaced as LoadBalanceStats::unmet_flux):
 * no exception, no silent log — the phenomenon stays measurable.
 *
 * Communication: neighbour-only point-to-point for the loads/degrees and the
 * neighbour meshes, + 1 boolean all_reduce per flux iteration + 1 scalar
 * all_reduce for the convergence scale. No O(P) gather of loads or meshes.
 *
 * Guarantees: connected cessions (no islands); convergence to balance over
 * several calls (each call sheds at most the available domain thickness);
 * compact partitions with staircase boundaries.
 *
 * Reference: G. Cybenko, "Dynamic load balancing for distributed memory
 * multiprocessors", J. Parallel Distrib. Comput. 7 (1989) 279-301.
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../../algorithm.hpp"
#include "../../field.hpp"
#include "../../subset/node.hpp"
#include "../config.hpp"
#include "../metrics.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    namespace detail
    {
        /// Symmetric scalar exchange with every neighbour rank (isend all, recv
        /// all). The process graph must be symmetric (if j lists me, I list j).
        template <class T>
        void exchange_scalar(const std::vector<int>& ranks, const T& mine, std::vector<T>& theirs)
        {
            namespace mpi = boost::mpi;
            mpi::communicator world;
            const std::size_t n = ranks.size();
            theirs.resize(n);

            std::vector<mpi::request> req;
            req.reserve(n);
            for (std::size_t j = 0; j < n; ++j)
            {
                req.push_back(world.isend(ranks[j], tag_diffusion, mine));
            }
            for (std::size_t j = 0; j < n; ++j)
            {
                world.recv(ranks[j], tag_diffusion, theirs[j]);
            }
            mpi::wait_all(req.begin(), req.end());
        }

        /**
         * Cybenko diffusion on the process graph. Given this process' load and
         * its neighbour ranks, returns one signed flux per neighbour: negative =
         * load to give, positive = load to receive. See the file header.
         *
         * @note MPI: neighbour-only point-to-point (degrees once, loads per
         *       iteration) + 1 scalar all_reduce (convergence scale) + 1 boolean
         *       all_reduce per iteration. No load gather.
         */
        inline std::vector<double> diffusion_fluxes(double my_load, const std::vector<int>& neighbour_ranks, const LoadBalanceConfig& config)
        {
            namespace mpi = boost::mpi;
            mpi::communicator world;

            const std::size_t n = neighbour_ranks.size();
            std::vector<double> fluxes(n, 0.);

            const double my_deg = static_cast<double>(n);
            std::vector<double> neigh_deg;
            exchange_scalar(neighbour_ranks, my_deg, neigh_deg);

            // convergence scale: a transfer below epsilon is negligible. One
            // scalar all_reduce sets the global average load (no load gather).
            const double w_total = mpi::all_reduce(world, my_load, std::plus<double>());
            const double avg     = (world.size() > 0) ? w_total / static_cast<double>(world.size()) : 0.;
            const double scale   = (avg > 0.) ? avg : 1.;
            const double epsilon = 1e-3 * scale;

            for (int iter = 0; iter < config.diffusion_iterations; ++iter)
            {
                std::vector<double> neigh_load;
                exchange_scalar(neighbour_ranks, my_load, neigh_load);

                double delta         = 0.;
                bool local_converged = true;
                for (std::size_t j = 0; j < n; ++j)
                {
                    const double t = (neigh_load[j] - my_load) / (std::max(my_deg, neigh_deg[j]) + 1.);
                    fluxes[j] += t;
                    delta += t;
                    if (std::abs(t) > epsilon)
                    {
                        local_converged = false;
                    }
                }
                my_load += delta;

                if (mpi::all_reduce(world, local_converged, std::logical_and<bool>()))
                {
                    break;
                }
            }

            // drop micro-fluxes (avoid migrating a handful of cells back and forth)
            for (std::size_t j = 0; j < n; ++j)
            {
                if (std::abs(fluxes[j]) < config.flux_threshold * scale)
                {
                    fluxes[j] = 0.;
                }
            }
            return fluxes;
        }
    }

    /**
     * Diffusion strategy (see the file header for the algorithm).
     *
     * Holds a `LoadBalanceConfig` (flux threshold, iteration count) and exposes
     * `last_unmet_flux()` so the driver can report the load it could not shed.
     */
    class Diffusion
    {
      public:

        Diffusion() = default;

        explicit Diffusion(LoadBalanceConfig config)
            : m_config(config)
        {
        }

        std::string name() const
        {
            return "diffusion";
        }

        /// Load this process wanted to shed but could not on the last call
        /// (interface exhausted). Read by the driver into LoadBalanceStats.
        double last_unmet_flux() const
        {
            return m_unmet_flux;
        }

        /**
         * Destination rank of each cell: keep everything, then cede interface
         * layers towards the neighbours the diffusion fluxes point to.
         *
         * @note MPI: neighbour-only point-to-point (neighbour meshes, loads,
         *       degrees) + 1 boolean all_reduce per flux iteration + 1 scalar
         *       all_reduce for the convergence scale. No gather.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            using cl_type   = typename Mesh::cl_type;
            using ca_type   = typename Mesh::ca_type;

            boost::mpi::communicator world;

            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());
            m_unmet_flux = 0.;

            auto& neighbourhood = mesh.mpi_neighbourhood();
            const std::size_t n = neighbourhood.size();
            if (n == 0)
            {
                return flags; // isolated subdomain: nothing to exchange
            }

            // The layer construction needs the neighbours' actual cells, which
            // the mesh keeps only as a bounding box after construction. This is
            // a neighbour-only point-to-point exchange.
            mesh.update_mesh_neighbour();

            // -- phase 1: fluxes (neighbour-only diffusion) -------------------------
            const std::vector<double> fluxes = compute_fluxes(mesh, weight, neighbourhood);

            // Process the biggest givers first (most negative flux), deterministic
            // tie-break on the neighbour rank.
            std::vector<std::size_t> order(n);
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::sort(order.begin(),
                      order.end(),
                      [&](std::size_t a, std::size_t b)
                      {
                          if (fluxes[a] != fluxes[b])
                          {
                              return fluxes[a] < fluxes[b];
                          }
                          return neighbourhood[a].rank < neighbourhood[b].rank;
                      });

            const auto bc_me = barycenter(mesh);

            // -- phase 2: cede interface layers, neighbour by neighbour -------------
            for (std::size_t idx : order)
            {
                if (fluxes[idx] >= 0.)
                {
                    break; // sorted ascending: the rest are receivers
                }
                double remaining = -fluxes[idx]; // load I must give to this neighbour

                const int neigh_rank = neighbourhood[idx].rank;
                const auto bc_j      = barycenter(neighbourhood[idx].mesh);
                const auto dirs      = cession_directions<Mesh::dim>(bc_me, bc_j);

                for (const auto& dir : dirs)
                {
                    if (remaining <= 0.)
                    {
                        break;
                    }
                    give_layers<cl_type, ca_type>(mesh, flags, neighbourhood[idx].mesh, dir, neigh_rank, weight, remaining);
                }

                if (remaining > 0.)
                {
                    m_unmet_flux += remaining; // interface exhausted: report the deficit
                }
            }

            return flags;
        }

      private:

        /// Diffusion fluxes for this process: the local weighted load fed into
        /// the pure solver `detail::diffusion_fluxes`. One signed value per
        /// neighbour (negative = give, positive = receive).
        template <class Mesh, class Weight, class Neighbourhood>
        std::vector<double> compute_fluxes(const Mesh& mesh, const Weight& weight, const Neighbourhood& neighbourhood) const
        {
            std::vector<int> ranks;
            ranks.reserve(neighbourhood.size());
            for (const auto& neigh : neighbourhood)
            {
                ranks.push_back(neigh.rank);
            }
            return detail::diffusion_fluxes(local_load(mesh, weight), ranks, m_config);
        }

        /// Geometric (unweighted) barycenter of a mesh's leaves. Used only to
        /// pick the cession direction, so weighting is irrelevant — and it lets
        /// us reuse it on a neighbour mesh, where a field-based weight would be
        /// out of range.
        template <class Mesh>
        static auto barycenter(const Mesh& mesh)
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            constexpr std::size_t dim = Mesh::dim;
            xt::xtensor_fixed<double, xt::xshape<dim>> bc;
            bc.fill(0.);
            double count = 0.;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              const auto center = cell.center();
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  bc(d) += center(d);
                              }
                              count += 1.;
                          });
            count = std::max(count, 1e-12);
            for (std::size_t d = 0; d < dim; ++d)
            {
                bc(d) /= count;
            }
            return bc;
        }

        /// Cardinal axes pointing from the neighbour towards me (i.e. into my
        /// domain): one unit vector per axis along which my barycenter is past
        /// the neighbour's. A face neighbour yields one axis, a corner several.
        template <std::size_t dim, class Coord>
        static auto cession_directions(const Coord& bc_me, const Coord& bc_neigh)
        {
            using direction_t = xt::xtensor_fixed<int, xt::xshape<dim>>;

            std::vector<direction_t> dirs;
            double norm = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                norm += (bc_me(d) - bc_neigh(d)) * (bc_me(d) - bc_neigh(d));
            }
            norm = std::sqrt(norm);
            if (norm < 1e-12)
            {
                return dirs;
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                const double comp = (bc_me(d) - bc_neigh(d)) / norm;
                int s             = static_cast<int>(comp / 0.5); // |comp| >= 0.5 -> +-1
                s                 = std::clamp(s, -1, 1);
                if (s != 0)
                {
                    direction_t dd;
                    dd.fill(0);
                    dd(d) = s;
                    dirs.push_back(dd);
                }
            }
            return dirs;
        }

        /**
         * Cede successive layers of my cells to `neigh_rank`, starting at the
         * interface with the neighbour and moving by `dir` into my domain, until
         * `remaining` load is shed or the interface is exhausted. `dir` is a unit
         * cardinal vector pointing from the neighbour into my domain.
         *
         * Everything is built at `min_level` and projected onto each real level
         * with `.on(level)`, which is dimension- and level-jump-agnostic.
         */
        template <class cl_type, class ca_type, class Mesh, class Flags, class NeighMesh, class Direction, class Weight>
        void give_layers(Mesh& mesh,
                         Flags& flags,
                         const NeighMesh& neigh_mesh,
                         const Direction& dir,
                         int neigh_rank,
                         const Weight& weight,
                         double& remaining) const
        {
            using mesh_id_t       = typename Mesh::mesh_id_t;
            const std::size_t ref = mesh.min_level();
            const int rank        = boost::mpi::communicator{}.rank();

            // my whole subdomain and the neighbour's cells, projected to ref
            const ca_type my_ref    = project_to_level<cl_type, ca_type>(mesh[mesh_id_t::cells], ref);
            const ca_type neigh_ref = project_to_level<cl_type, ca_type>(neigh_mesh[mesh_id_t::cells], ref);

            // layer 0: my ref cells one step `dir` away from the neighbour, i.e.
            // the neighbour cells brought onto my side intersected with mine.
            cl_type if0_cl;
            intersection(my_ref[ref], translate(neigh_ref[ref], dir))(
                [&](const auto& interval, const auto& index)
                {
                    if0_cl[ref][index].add_interval(interval);
                });
            const ca_type interface = {if0_cl, false};
            if (interface[ref].empty())
            {
                return;
            }

            for (int offset = 0; remaining > 0.; ++offset)
            {
                auto band        = translate(interface[ref], dir * offset); // ref-level slab
                std::size_t gave = 0;

                for (std::size_t level = mesh.min_level(); level <= mesh.max_level() && remaining > 0.; ++level)
                {
                    intersection(band, mesh[mesh_id_t::cells][level])
                        .on(level)(
                            [&](const auto& interval, const auto& index)
                            {
                                for (auto i = interval.start; i < interval.end && remaining > 0.; ++i)
                                {
                                    auto cell = mesh.get_cell(level, i, index);
                                    if (flags[cell] == rank)
                                    {
                                        flags[cell] = neigh_rank;
                                        remaining -= weight(cell);
                                        ++gave;
                                    }
                                }
                            });
                }

                if (gave == 0)
                {
                    break; // nothing left connected to the interface in this direction
                }
            }
        }

        /// Project all levels of `cells` down to `ref` (coarsening), returning a
        /// single-level CellArray. `ref` must be <= every level of `cells`.
        template <class cl_type, class ca_type, class CellArray>
        static ca_type project_to_level(const CellArray& cells, std::size_t ref)
        {
            cl_type cl;
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                self(cells[level])
                    .on(ref)(
                        [&](const auto& interval, const auto& index)
                        {
                            cl[ref][index].add_interval(interval);
                        });
            }
            return ca_type{cl, false};
        }

        LoadBalanceConfig m_config{};
        mutable double m_unmet_flux = 0.;
    };
}
#endif
