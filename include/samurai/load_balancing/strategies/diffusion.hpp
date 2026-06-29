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
 *  2. Layer assignment (nD), by a geometric breadth-first peel. A negative flux
 *     fluxes[j] means "I must shed |fluxes[j]| of load to neighbour j". We grow
 *     the ceded region OUT OF THE ACTUAL INTERFACE with j: the first layer is my
 *     cells face-adjacent to j's cells, the next layer is my still-owned cells
 *     face-adjacent to what I just ceded, and so on (a BFS front advancing into
 *     my domain). Crucially the front only ever crosses cells I still own, so:
 *       - the ceded region stays attached to j (it is connected to j's side);
 *       - what remains mine stays a single connected island — the peel never
 *         jumps across the domain, because adjacency is recomputed from the
 *         cells just given, not from a fixed Cartesian direction.
 *     There is therefore no notion of direction at all (the previous
 *     barycentre-direction version scattered cells on adaptive meshes and broke
 *     connectivity). Adjacency is evaluated at the coarsest level (`min_level`)
 *     by set algebra over the 2*dim cardinal translations and projected onto
 *     every actual level with `.on(level)`, which is dimension- and level-jump
 *     agnostic. The frontier between subdomains is a staircase, not a straight
 *     line (the straight-line / row-snapping constraint limited the old version
 *     to 2D bands).
 *
 *     The peel is ATOMIC at `min_level`: a coarse cell is ceded with ALL the
 *     fine cells it contains, or not at all, and we stop at a coarse-cell
 *     boundary once the requested flux is met. This is essential on adaptive
 *     meshes: stopping mid-cell at the finest level (a raw cell scan) used to
 *     dice the refined front into disconnected single-cell slivers — the very
 *     islands this peel is meant to avoid. The cost is an overshoot bounded by
 *     one coarse cell's load; it is small when refinement tracks an interface
 *     (a coarse cell then holds only a thin band of fine cells), and the
 *     diffusion converges over several calls anyway.
 *
 *  3. Connectivity repair. Shedding to several neighbours whose territories
 *     wrap around a process can still split the cells it keeps into
 *     disconnected pockets (it happens in 3D, where a subdomain has more
 *     neighbours). A final pass labels the kept region's connected components
 *     (seed-growth flood fill at `min_level`) and hands every pocket but the
 *     largest to the neighbour that borders it most — restoring a single
 *     connected island per process. The pass is a no-op (one flood fill) when
 *     the kept region is already connected, i.e. on almost every call.
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
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../../algorithm.hpp"
#include "../../field.hpp"
#include "../../stencil.hpp"
#include "../../subset/node.hpp"
#include "../config.hpp"
#include "../metrics.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    /**
     * Options for the diffusion strategy. Mirrors the per-strategy option
     * structs of the module (`MetisOptions`, `ScotchOptions`) so the strategy's
     * tunables stay separate from the driver's `LoadBalanceConfig`.
     */
    struct DiffusionOptions
    {
        /// Fluxes smaller than this fraction of the local average load are
        /// zeroed out to avoid micro-migrations.
        double flux_threshold = 0.01;

        /// Maximum number of iterations of the iterative flux solver.
        int diffusion_iterations = 50;

        /// Fraction of its weighted load a process always keeps: the total it
        /// sheds in one call is capped at `(1 - min_retained_load_fraction) *
        /// local_load`. A process cannot give away more cells than it owns, and
        /// the iterative flux solver may transiently ask for more than the whole
        /// load (it oscillates before converging, especially with many
        /// processes); without this floor a process could empty itself, which
        /// breaks the subsequent mesh adaptation. Keeping a small reserve
        /// guarantees a non-empty, valid subdomain every call.
        double min_retained_load_fraction = 0.1;
    };

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
        inline std::vector<double> diffusion_fluxes(double my_load, const std::vector<int>& neighbour_ranks, const DiffusionOptions& options)
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

            for (int iter = 0; iter < options.diffusion_iterations; ++iter)
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
                if (std::abs(fluxes[j]) < options.flux_threshold * scale)
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
     * Holds its `DiffusionOptions` (flux threshold, iteration count, retained
     * load fraction) and exposes `last_unmet_flux()` so the driver can report
     * the load it could not shed.
     */
    class Diffusion
    {
      public:

        Diffusion() = default;

        explicit Diffusion(DiffusionOptions options)
            : m_options(options)
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
            using cl_type = typename Mesh::cl_type;
            using ca_type = typename Mesh::ca_type;

            boost::mpi::communicator world;

            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());
            m_unmet_flux = 0.;

            auto& neighbourhood = mesh.mpi_neighbourhood();
            const std::size_t n = neighbourhood.size();

            // The layer construction needs the neighbours' actual cells, which
            // the mesh keeps only as a bounding box after construction. This is
            // a neighbour-only point-to-point exchange -- nothing to do, and
            // safe to skip, when this rank has no neighbour.
            if (n > 0)
            {
                mesh.update_mesh_neighbour();
            }

            // -- phase 1: fluxes (neighbour-only diffusion) -------------------------
            // EVERY rank must reach the flux solver, even one with no MPI
            // neighbour: diffusion_fluxes runs world-collective all_reduce calls
            // (the global load once, the convergence flag every iteration). A
            // rank that returned early on n == 0 would skip those collectives and
            // deadlock every rank that still has a neighbour -- this happens once
            // the partition isolates a subdomain (e.g. many ranks on a thin
            // domain). With an empty neighbour list the solver only joins the
            // all_reduce and returns no flux, so the peel below is simply empty.
            std::vector<double> fluxes = compute_fluxes(mesh, weight, neighbourhood);

            if (n == 0)
            {
                return flags; // isolated subdomain: collectives joined, nothing to peel
            }

            // A process cannot shed more cells than it owns. The iterative flux
            // solver may transiently ask for more than the whole load (it
            // oscillates before converging, especially with many processes), so
            // we cap the total cession and keep a small reserve: this guarantees
            // a non-empty subdomain, which the subsequent mesh adaptation needs.
            double total_give = 0.;
            for (const double f : fluxes)
            {
                if (f < 0.)
                {
                    total_give -= f;
                }
            }
            const double max_give = (1. - m_options.min_retained_load_fraction) * local_load(mesh, weight);
            if (total_give > max_give && total_give > 0.)
            {
                const double scale = max_give / total_give;
                for (double& f : fluxes)
                {
                    if (f < 0.)
                    {
                        f *= scale;
                    }
                }
            }

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

            // -- phase 2: peel interface layers, neighbour by neighbour ------------
            for (std::size_t idx : order)
            {
                if (fluxes[idx] >= 0.)
                {
                    break; // sorted ascending: the rest are receivers
                }
                double remaining = -fluxes[idx]; // load I must give to this neighbour

                give_to_neighbour<cl_type, ca_type>(mesh, flags, neighbourhood[idx].mesh, neighbourhood[idx].rank, weight, remaining);

                if (remaining > 0.)
                {
                    m_unmet_flux += remaining; // interface exhausted: report the deficit
                }
            }

            // Shedding to several neighbours that wrap around me can split the
            // cells I keep into disconnected pockets; relabel the secondary ones
            // so every subdomain stays in one piece (see repair_connectivity).
            repair_connectivity<cl_type, ca_type>(mesh, flags);

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
            return detail::diffusion_fluxes(local_load(mesh, weight), ranks, m_options);
        }

        /**
         * Cede load to `neigh_rank` by a geometric breadth-first peel that keeps
         * both the ceded region and my remaining region connected (see the file
         * header). Adjacency is computed at `min_level`; a coarse cell is ceded
         * by handing every still-owned fine cell it contains to the neighbour.
         *
         * `boundary` is the advancing front (the neighbour's region plus what I
         * have already given); `owned` shrinks as I peel, and the next ring is
         * always `owned ∩ (cells face-adjacent to boundary)`, so the front never
         * leaves the cells I own — no island is ever created.
         */
        template <class cl_type, class ca_type, class Mesh, class Flags, class NeighMesh, class Weight>
        void
        give_to_neighbour(Mesh& mesh, Flags& flags, const NeighMesh& neigh_mesh, int neigh_rank, const Weight& weight, double& remaining) const
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            constexpr std::size_t dim = Mesh::dim;
            const std::size_t ref     = mesh.min_level();
            const int rank            = boost::mpi::communicator{}.rank();

            // my still-owned cells (flags == me), at their own levels then
            // projected to the coarsest level for the adjacency computation
            cl_type owned_full_cl;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              if (flags[cell] != rank)
                              {
                                  return;
                              }
                              auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                              owned_full_cl[cell.level][yz].add_point(cell.indices[0]);
                          });
            const ca_type owned_full = {owned_full_cl, false};
            ca_type owned            = project_to_level<cl_type, ca_type>(owned_full, ref);

            // the BFS front, seeded with the neighbour's region at the coarse level
            ca_type boundary = project_to_level<cl_type, ca_type>(neigh_mesh[mesh_id_t::cells], ref);

            const auto directions = cartesian_directions<dim>();

            while (remaining > 0.)
            {
                // ring = my owned coarse cells face-adjacent to the current front
                cl_type ring_cl;
                for (std::size_t k = 0; k < directions.shape()[0]; ++k)
                {
                    auto d = xt::view(directions, k);
                    intersection(owned[ref], translate(boundary[ref], d))(
                        [&](const auto& interval, const auto& index)
                        {
                            ring_cl[ref][index].add_interval(interval);
                        });
                }
                ca_type ring = {ring_cl, false};
                if (ring[ref].empty())
                {
                    break; // front cannot advance without leaving my owned cells
                }

                // Cede the ring ONE COARSE CELL AT A TIME, atomically: a coarse
                // cell (at `ref` = min_level) is either fully handed to the
                // neighbour or fully kept. This is what keeps the partition
                // compact. Ceding whole rings and stopping mid-ring at the finest
                // level — a raw cell scan — used to slice the refined front into
                // disconnected single-cell slivers (the islands seen on adaptive
                // meshes). With atomic coarse cells the frontier is a min_level
                // staircase, and since every coarse cell of the ring is
                // face-adjacent to the front, everything we cede stays connected
                // to the neighbour: no island can appear. The price is an
                // overshoot bounded by one coarse cell's worth of load — small
                // when refinement is along an interface (a coarse cell then holds
                // only a thin band of fine cells).
                cl_type ceded_cl;
                bool stop = false;
                for_each_interval(ring[ref],
                                  [&](std::size_t /*lvl*/, const auto& interval, const auto& index)
                                  {
                                      for (auto i = interval.start; i < interval.end && !stop; ++i)
                                      {
                                          // hand every still-owned fine cell of this one coarse cell
                                          cl_type cc_cl;
                                          cc_cl[ref][index].add_point(i);
                                          const ca_type cc = {cc_cl, false};

                                          for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                                          {
                                              intersection(cc[ref], mesh[mesh_id_t::cells][level])
                                                  .on(level)(
                                                      [&](const auto& fine_interval, const auto& fine_index)
                                                      {
                                                          for (auto fi = fine_interval.start; fi < fine_interval.end; ++fi)
                                                          {
                                                              auto cell = mesh.get_cell(level, fi, fine_index);
                                                              if (flags[cell] == rank)
                                                              {
                                                                  flags[cell] = neigh_rank;
                                                                  remaining -= weight(cell);
                                                              }
                                                          }
                                                      });
                                          }

                                          ceded_cl[ref][index].add_point(i);
                                          if (remaining <= 0.)
                                          {
                                              stop = true; // requested flux met: stop at this coarse-cell boundary
                                          }
                                      }
                                  });

                // advance the front onto the coarse cells actually ceded, and
                // drop them from the owned set (a partially-peeled ring leaves
                // its remaining coarse cells owned, for the next neighbour or me)
                const ca_type ceded = {ceded_cl, false};
                boundary            = set_union<cl_type, ca_type>(boundary, ceded, ref);
                owned               = set_difference<cl_type, ca_type>(owned, ceded, ref);

                if (stop)
                {
                    break;
                }
            }
        }

        /**
         * Repair the connectivity of the kept region. The per-neighbour peel
         * cedes regions each connected to their neighbour, but when a process
         * sheds to several neighbours whose territories wrap around it, the
         * cells it keeps can split into disconnected pockets (this happens in
         * 3D, where a subdomain has more neighbours). Each secondary pocket is
         * fully surrounded by cells already promised to neighbours, so we hand
         * it to the neighbour that borders it most: the pocket is connected and
         * adjacent to that neighbour, hence both subdomains stay in one piece.
         *
         * Connected components are found by a seed-growth flood fill at
         * `min_level` (set algebra over the cardinal translations, no O(n^2)
         * cell scan); the largest component is kept. The pass is a no-op — one
         * flood fill that converges to the whole region — when the kept region
         * is already connected, i.e. on the vast majority of calls.
         */
        template <class cl_type, class ca_type, class Mesh, class Flags>
        void repair_connectivity(Mesh& mesh, Flags& flags) const
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            constexpr std::size_t dim = Mesh::dim;
            const std::size_t ref     = mesh.min_level();
            const int rank            = boost::mpi::communicator{}.rank();
            const auto directions     = cartesian_directions<dim>();

            // my kept cells (flags == me), projected to the coarsest level
            cl_type kept_cl;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              if (flags[cell] != rank)
                              {
                                  return;
                              }
                              auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                              kept_cl[cell.level][yz].add_point(cell.indices[0]);
                          });
            const ca_type kept = project_to_level<cl_type, ca_type>(ca_type{kept_cl, false}, ref);
            if (kept[ref].empty())
            {
                return;
            }

            // enumerate connected components by seed growth (BFS at ref)
            std::vector<ca_type> components;
            ca_type remaining = kept;
            while (!remaining[ref].empty())
            {
                cl_type seed_cl;
                bool placed = false;
                for_each_interval(remaining[ref],
                                  [&](std::size_t /*lvl*/, const auto& interval, const auto& index)
                                  {
                                      if (!placed)
                                      {
                                          seed_cl[ref][index].add_point(interval.start);
                                          placed = true;
                                      }
                                  });
                ca_type comp = {seed_cl, false};
                for (;;)
                {
                    cl_type grow_cl;
                    for (std::size_t k = 0; k < directions.shape()[0]; ++k)
                    {
                        auto d = xt::view(directions, k);
                        intersection(remaining[ref], translate(comp[ref], d))(
                            [&](const auto& interval, const auto& index)
                            {
                                grow_cl[ref][index].add_interval(interval);
                            });
                    }
                    const ca_type grown = set_union<cl_type, ca_type>(comp, ca_type{grow_cl, false}, ref);
                    if (grown[ref].nb_cells() == comp[ref].nb_cells())
                    {
                        break; // no new neighbour reached: component complete
                    }
                    comp = grown;
                }
                components.push_back(comp);
                remaining = set_difference<cl_type, ca_type>(remaining, comp, ref);
            }

            if (components.size() <= 1)
            {
                return; // already a single connected island
            }

            // keep the largest component, relabel every other one
            std::size_t main = 0;
            for (std::size_t c = 1; c < components.size(); ++c)
            {
                if (components[c][ref].nb_cells() > components[main][ref].nb_cells())
                {
                    main = c;
                }
            }

            // my cells already ceded to a neighbour (mine \ kept), at the coarse level
            const ca_type mine  = project_to_level<cl_type, ca_type>(mesh[mesh_id_t::cells], ref);
            const ca_type ceded = set_difference<cl_type, ca_type>(mine, kept, ref);

            for (std::size_t c = 0; c < components.size(); ++c)
            {
                if (c == main)
                {
                    continue;
                }
                const ca_type& comp = components[c];

                // ceded coarse cells touching this pocket
                cl_type border_cl;
                for (std::size_t k = 0; k < directions.shape()[0]; ++k)
                {
                    auto d = xt::view(directions, k);
                    intersection(translate(comp[ref], d), ceded[ref])(
                        [&](const auto& interval, const auto& index)
                        {
                            border_cl[ref][index].add_interval(interval);
                        });
                }
                const ca_type border = {border_cl, false};

                // vote: which neighbour owns the most of the pocket's border
                std::map<int, std::size_t> votes;
                for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                {
                    intersection(border[ref], mesh[mesh_id_t::cells][level])
                        .on(level)(
                            [&](const auto& interval, const auto& index)
                            {
                                for (auto i = interval.start; i < interval.end; ++i)
                                {
                                    const int f = flags[mesh.get_cell(level, i, index)];
                                    if (f != rank)
                                    {
                                        ++votes[f];
                                    }
                                }
                            });
                }
                if (votes.empty())
                {
                    continue; // no neighbour borders it (rare): leave it with me
                }
                int target       = votes.begin()->first;
                std::size_t best = 0;
                for (const auto& [r, v] : votes)
                {
                    if (v > best)
                    {
                        best   = v;
                        target = r;
                    }
                }

                // hand the whole pocket to that neighbour
                for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                {
                    intersection(comp[ref], mesh[mesh_id_t::cells][level])
                        .on(level)(
                            [&](const auto& interval, const auto& index)
                            {
                                for (auto i = interval.start; i < interval.end; ++i)
                                {
                                    auto cell = mesh.get_cell(level, i, index);
                                    if (flags[cell] == rank)
                                    {
                                        flags[cell] = target;
                                    }
                                }
                            });
                }
            }
        }

        /// Materialize `a ∪ b` at level `ref` into a single-level CellArray.
        template <class cl_type, class ca_type, class CellArray>
        static ca_type set_union(const CellArray& a, const CellArray& b, std::size_t ref)
        {
            cl_type cl;
            union_(a[ref], b[ref])(
                [&](const auto& interval, const auto& index)
                {
                    cl[ref][index].add_interval(interval);
                });
            return ca_type{cl, false};
        }

        /// Materialize `a \ b` at level `ref` into a single-level CellArray.
        template <class cl_type, class ca_type, class CellArray>
        static ca_type set_difference(const CellArray& a, const CellArray& b, std::size_t ref)
        {
            cl_type cl;
            difference(a[ref], b[ref])(
                [&](const auto& interval, const auto& index)
                {
                    cl[ref][index].add_interval(interval);
                });
            return ca_type{cl, false};
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

        DiffusionOptions m_options{};
        mutable double m_unmet_flux = 0.;
    };
}
#endif
