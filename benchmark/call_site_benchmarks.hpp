// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The benchmarks of the call sites (set_call_sites.hpp) and of the ways to
// consume a set (set_modes.hpp), for one backend of the set algebra
// (set_algebra_backend.hpp). register_call_sites<B, dim> registers them as
// <B::name>/BM_Site_X<dim> and <B::name>/BM_Mode_X<dim>, so that
// --benchmark_filter='^<name>/' selects the backend; register_partition_sites
// registers those of an MPI rank on the partitions of a mesh.
//
// Args: (max_level, number of balls, stencil radius, geometry), see
// bench::mesh_cases, and the number of ranks for a partition, see
// bench::partition_cases.
//
// Each site is written once against a context, which gives its inputs and
// runs it: Timing runs it in a benchmark, Recording computes its checksum
// for generate_set_algebra_inputs. Before the timed loop, Timing computes
// the checksum of the backend once and compares it with the reference the
// generator wrote (site_checksum.hpp); on a mismatch the benchmark stops
// with an error and bench::main returns a failure.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_backend.hpp"
#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
#include "set_modes.hpp"
#include "site_checksum.hpp"
#include "site_inputs.hpp"

namespace bench
{
    // The result of the check of each benchmark, by name: empty if the
    // backend computed the reference sets, the mismatch otherwise. A
    // benchmark runs several times (repetitions, iteration estimates); it is
    // checked once.
    inline std::map<std::string, std::string>& verifications()
    {
        static std::map<std::string, std::string> results;
        return results;
    }

    inline std::size_t verification_failures()
    {
        return static_cast<std::size_t>(std::count_if(verifications().begin(),
                                                      verifications().end(),
                                                      [](const auto& result)
                                                      {
                                                          return !result.second.empty();
                                                      }));
    }

    // The context of a site in a benchmark of backend B: its inputs come
    // from the files, and it runs in the timed loop once its checksum
    // matches the reference.
    template <class B, std::size_t dim>
    class Timing
    {
      public:

        Timing(benchmark::State& state, std::string site, bool partition)
            : m_state(state)
            , m_site(std::move(site))
            , m_case(partition ? partition_case(state) : mesh_case(state))
        {
        }

        const backend_inputs_t<B, dim>& inputs(bool moved = false)
        {
            return backend_inputs<B, dim>(m_state, moved);
        }

        const std::vector<backend_inputs_t<B, dim>>& partition()
        {
            return backend_partition<B, dim>(m_state);
        }

        // site(f) passes each set it evaluates to f(level, set), which
        // returns its number of intervals. The callback only counts: the
        // time is that of the set algebra in the library function, to set
        // against BM_SiteOp_X or BM_Op_* of bench_set_library_ops.
        template <class Site>
        void traverse(Site&& site)
        {
            if (verify(
                    [&]
                    {
                        return checksum_traverse_site<B, dim>(site);
                    }))
            {
                m_state.counters["sets"] = static_cast<double>(run_site<B>(m_state, site));
            }
        }

        // site(emit) builds level cell arrays and passes each one it keeps
        // to emit(level, lca): the time is that of building them, `itv` and
        // `cells` count what they keep.
        template <class Site>
        void build(Site&& site)
        {
            if (verify(
                    [&]
                    {
                        return checksum_build_site<B, dim>(site);
                    }))
            {
                m_state.counters["sets"] = static_cast<double>(run_build_site<B>(m_state, site));
            }
        }

        // site(f) passes each set it evaluates to f(level, set), which only
        // asks whether it is empty.
        template <class Site>
        void empty(Site&& site)
        {
            if (verify(
                    [&]
                    {
                        return checksum_empty_site<B>(site);
                    }))
            {
                m_state.counters["sets"] = static_cast<double>(run_empty_site<B>(m_state, site));
            }
        }

      private:

        // The name of the benchmark, without the backend.
        std::string name() const
        {
            std::string name = m_site + "<" + std::to_string(dim) + ">";
            for (const auto arg : {m_case.max_level, m_case.nballs, static_cast<std::size_t>(m_case.stencil_radius)})
            {
                name += "/" + std::to_string(arg);
            }
            name += "/" + std::to_string(static_cast<int>(m_case.geometry));
            if (m_case.ranks > 0)
            {
                name += "/" + std::to_string(m_case.ranks);
            }
            return name;
        }

        template <class Checksum>
        bool verify(Checksum&& checksum)
        {
            const std::string key = std::string(B::name) + "/" + name();
            auto it               = verifications().find(key);
            if (it == verifications().end())
            {
                std::string mismatch;
                try
                {
                    mismatch = checksum_mismatch(checksum(), reference_checksum<dim>(m_case, m_site));
                }
                catch (const std::exception& e)
                {
                    mismatch = e.what();
                }
                if (!mismatch.empty())
                {
                    mismatch = "not the sets of the reference: " + mismatch;
                    std::cerr << key << ": " << mismatch << std::endl;
                }
                it = verifications().emplace(key, std::move(mismatch)).first;
            }
            if (!it->second.empty())
            {
                m_state.SkipWithError(it->second);
                return false;
            }
            return true;
        }

        benchmark::State& m_state;
        std::string m_site;
        MeshCase m_case;
    };

    // The context of a site in generate_set_algebra_inputs: runs it once on
    // the inputs it is given and keeps its checksum.
    template <class B, std::size_t dim>
    class Recording
    {
      public:

        using inputs_t = backend_inputs_t<B, dim>;

        // A whole mesh and the same mesh moved by one step.
        Recording(const inputs_t& in, const inputs_t& moved)
            : m_inputs(&in)
            , m_moved(&moved)
        {
        }

        // The ranks of a partition.
        explicit Recording(const std::vector<inputs_t>& ranks)
            : m_partition(&ranks)
        {
        }

        const inputs_t& inputs(bool moved = false) const
        {
            const auto* in = moved ? m_moved : m_inputs;
            if (in == nullptr)
            {
                throw std::logic_error("a partition site reads a whole mesh");
            }
            return *in;
        }

        const std::vector<inputs_t>& partition() const
        {
            if (m_partition == nullptr)
            {
                throw std::logic_error("a site of a whole mesh reads a partition");
            }
            return *m_partition;
        }

        template <class Site>
        void traverse(Site&& site)
        {
            m_checksum = checksum_traverse_site<B, dim>(site);
        }

        template <class Site>
        void build(Site&& site)
        {
            m_checksum = checksum_build_site<B, dim>(site);
        }

        template <class Site>
        void empty(Site&& site)
        {
            m_checksum = checksum_empty_site<B>(site);
        }

        const SiteChecksum& checksum() const
        {
            return m_checksum;
        }

      private:

        const inputs_t* m_inputs                 = nullptr;
        const inputs_t* m_moved                  = nullptr;
        const std::vector<inputs_t>* m_partition = nullptr;
        SiteChecksum m_checksum;
    };

    // The sites that need more than their inputs. Each takes the context
    // ctx of the backend B.
    namespace call_sites
    {
        template <class B, class Ctx>
        void MeshPredictionGhosts(Ctx& ctx)
        {
            const auto& in = ctx.inputs();
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::mesh_prediction_ghosts<B>(in, sites::projected_subdomain<B>(in), f);
                });
        }

        // The Cached variants read the coarse domain or subdomain the mesh
        // already stores instead of coarsening the finest one at every use:
        // the gap with the site is what caching coarse sets would save.
        template <class B, class Ctx>
        void CachedMeshPredictionGhosts(Ctx& ctx)
        {
            const auto& in = ctx.inputs();
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::mesh_prediction_ghosts<B>(in, sites::cached_subdomain(in), f);
                });
        }

        template <class B, class Ctx>
        void OuterGhosts(Ctx& ctx)
        {
            const auto& in = ctx.inputs();
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::outer_ghosts<B>(in, sites::projected_domain<B>(in), f);
                });
        }

        template <class B, class Ctx>
        void CachedOuterGhosts(Ctx& ctx)
        {
            const auto& in = ctx.inputs();
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::outer_ghosts<B>(in, sites::cached_domain(in), f);
                });
        }

        template <class B, class Ctx>
        void FieldTransfer(Ctx& ctx)
        {
            const auto& in     = ctx.inputs();
            const auto& new_in = ctx.inputs(true);
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::field_transfer<B>(in, new_in, f);
                });
        }

        // The regions are built once, when the boundary condition is.
        template <class B, class Ctx>
        void BcCells(Ctx& ctx)
        {
            const auto& in     = ctx.inputs();
            const auto regions = sites::bc_regions<B>(in);
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::bc_cells<B>(in, regions, f);
                });
        }

        // The ghosts without an owner are found outside the timed loop.
        template <class B, class Ctx>
        void OwnershipPerCell(Ctx& ctx)
        {
            const auto& in    = ctx.inputs();
            const auto ghosts = sites::ownership_ghosts<B>(in);
            ctx.traverse(
                [&](auto&& f)
                {
                    sites::ownership_per_cell<B>(in, ghosts, f);
                });
        }

        // The sites on a partition run every rank in turn: the time is that
        // of all the ranks, as if one process ran them all.
        template <class Ranks>
        auto partition_neighbours(const Ranks& ranks)
        {
            std::vector<decltype(sites::neighbours_of(ranks, 0))> neighbours;
            for (std::size_t r = 0; r < ranks.size(); ++r)
            {
                neighbours.push_back(sites::neighbours_of(ranks, r));
            }
            return neighbours;
        }

        template <class B, class Ctx>
        void PartitionFindNeighbourhood(Ctx& ctx)
        {
            const auto& ranks = ctx.partition();
            ctx.empty(
                [&](auto&& f)
                {
                    for (std::size_t r = 0; r < ranks.size(); ++r)
                    {
                        auto found = sites::find_neighbourhood<B>(ranks, r, f);
                        benchmark::DoNotOptimize(found);
                    }
                });
        }

        template <class B, class Ctx>
        void PartitionConstructSubdomain(Ctx& ctx)
        {
            const auto& ranks = ctx.partition();
            ctx.build(
                [&](auto&& emit)
                {
                    for (const auto& in : ranks)
                    {
                        sites::construct_subdomain<B>(in, emit);
                    }
                });
        }

        // Every rank peels towards each of its neighbours, from the partition
        // as it is (see sites::diffusion_peel).
        template <class B, class Ctx>
        void PartitionDiffusionPeel(Ctx& ctx)
        {
            const auto& ranks     = ctx.partition();
            const auto neighbours = partition_neighbours(ranks);
            std::vector<double> loads;
            for (const auto& in : ranks)
            {
                loads.push_back(sites::peel_load<B>(in));
            }
            ctx.build(
                [&](auto&& emit)
                {
                    for (std::size_t r = 0; r < ranks.size(); ++r)
                    {
                        for (const auto* neighbour : neighbours[r])
                        {
                            sites::diffusion_peel<B>(ranks[r], *neighbour, loads[r], emit);
                        }
                    }
                });
        }
    } // namespace call_sites

    // Calls v(name, periodic_only, run) for each site of a whole mesh and
    // each mode, where run(ctx) runs it in the context ctx (Timing or
    // Recording of B in dim). periodic_only: the site has nothing to do on
    // a mesh that is not periodic.
    template <class B, std::size_t dim, class V>
    void for_each_call_site(V&& v)
    {
        // A site or mode that only reads the inputs of the mesh: KIND is
        // traverse, build or empty (see Timing).
#define BENCH_SIMPLE(NAME, KIND, FUNCTION, PERIODIC_ONLY) \
    v(NAME,                                               \
      PERIODIC_ONLY,                                      \
      [](auto& ctx)                                       \
      {                                                   \
          const auto& in = ctx.inputs();                  \
          ctx.KIND(                                       \
              [&](auto&& f)                               \
              {                                           \
                  FUNCTION<B>(in, f);                     \
              });                                         \
      })
#define BENCH_SITE(NAME, FUNCTION) \
    v(NAME,                        \
      false,                       \
      [](auto& ctx)                \
      {                            \
          FUNCTION<B>(ctx);        \
      })
        BENCH_SITE("BM_Site_MeshPredictionGhosts", call_sites::MeshPredictionGhosts);
        BENCH_SITE("BM_SiteCached_MeshPredictionGhosts", call_sites::CachedMeshPredictionGhosts);
        BENCH_SIMPLE("BM_Site_Corners", traverse, sites::corners, false);
        BENCH_SIMPLE("BM_Site_GhostProjection", traverse, sites::ghost_projection, false);
        BENCH_SIMPLE("BM_Site_GhostPrediction", traverse, sites::ghost_prediction, false);
        BENCH_SITE("BM_Site_OuterGhosts", call_sites::OuterGhosts);
        BENCH_SITE("BM_SiteCached_OuterGhosts", call_sites::CachedOuterGhosts);
        BENCH_SIMPLE("BM_Site_PeriodicGhosts", traverse, sites::periodic_ghosts, true);
        BENCH_SIMPLE("BM_Site_Detail", traverse, sites::detail, false);
        BENCH_SIMPLE("BM_Site_Tag", traverse, sites::tag, false);
        BENCH_SITE("BM_Site_FieldTransfer", call_sites::FieldTransfer);
        BENCH_SIMPLE("BM_Site_Interfaces", traverse, sites::interfaces, false);
        BENCH_SIMPLE("BM_Site_MeshSubMesh", build, sites::mesh_sub_mesh, false);
        BENCH_SIMPLE("BM_Site_ConstructUnion", build, sites::construct_union, false);
        BENCH_SIMPLE("BM_Site_BuildPyramid", build, sites::build_pyramid, false);
        BENCH_SIMPLE("BM_Site_CellArrayFromTag", build, sites::cell_array_from_tag, false);
        BENCH_SIMPLE("BM_Site_AmrSubMesh", build, sites::amr_sub_mesh, false);
        BENCH_SITE("BM_Site_BcCells", call_sites::BcCells);
        BENCH_SIMPLE("BM_Site_BcCorners", traverse, sites::bc_corners, false);
        BENCH_SIMPLE("BM_Site_BcFurtherGhosts", traverse, sites::bc_further_ghosts, false);
        BENCH_SIMPLE("BM_Site_PerCellChildren", traverse, sites::per_cell_children, false);
        BENCH_SITE("BM_Site_OwnershipPerCell", call_sites::OwnershipPerCell);

        BENCH_SIMPLE("BM_Mode_Traverse", traverse, modes::traverse_proj_cells, false);
        BENCH_SIMPLE("BM_Mode_Materialize", build, modes::materialize_proj_cells, false);
        BENCH_SIMPLE("BM_Mode_Build", build, modes::build_proj_cells, false);
        BENCH_SIMPLE("BM_Mode_EmptyHit", empty, modes::empty_hit, false);
        BENCH_SIMPLE("BM_Mode_EmptyMiss", empty, modes::empty_miss, false);
        BENCH_SIMPLE("BM_Mode_Contract", traverse, modes::contract, false);
        BENCH_SIMPLE("BM_Mode_ContractMasked", traverse, modes::contract_masked, false);
        BENCH_SIMPLE("BM_Mode_ExpandUniform", traverse, modes::expand_uniform, false);
        BENCH_SIMPLE("BM_Mode_ExpandAxes", traverse, modes::expand_axes, false);
        BENCH_SIMPLE("BM_Mode_ExpandMasked", traverse, modes::expand_masked, false);
        BENCH_SIMPLE("BM_Mode_OnMixedLevels", traverse, modes::on_mixed_levels, false);
        BENCH_SIMPLE("BM_Mode_OnSameLevel", traverse, modes::on_same_level, false);
        BENCH_SIMPLE("BM_Mode_NestedOn", traverse, modes::nested_on, false);
        BENCH_SIMPLE("BM_Mode_FlatOn", traverse, modes::flat_on, false);
        BENCH_SIMPLE("BM_Mode_IntersectionSelfOn", traverse, modes::intersection_self_on, false);
        BENCH_SIMPLE("BM_Mode_SelfOn", traverse, modes::self_on, false);
#undef BENCH_SIMPLE
#undef BENCH_SITE
    }

    // The same for the sites of an MPI rank on a partition: those that only
    // run with MPI, and the variants with the neighbours of the sites that
    // have one.
    template <class B, std::size_t dim, class V>
    void for_each_partition_site(V&& v)
    {
        const auto site = [&](const char* name, auto run, bool periodic_only = false)
        {
            v(name, periodic_only, run);
        };
        site("BM_Site_FindNeighbourhood",
             [](auto& ctx)
             {
                 call_sites::PartitionFindNeighbourhood<B>(ctx);
             });
        site("BM_Site_ConstructSubdomain",
             [](auto& ctx)
             {
                 call_sites::PartitionConstructSubdomain<B>(ctx);
             });
        // A site that each rank runs with its neighbours: KIND is traverse
        // or build (see Timing).
#define BENCH_PER_RANK(NAME, KIND, FUNCTION, PERIODIC_ONLY)                  \
    site(                                                                    \
        NAME,                                                                \
        [](auto& ctx)                                                        \
        {                                                                    \
            const auto& ranks     = ctx.partition();                         \
            const auto neighbours = call_sites::partition_neighbours(ranks); \
            ctx.KIND(                                                        \
                [&](auto&& f)                                                \
                {                                                            \
                    for (std::size_t r = 0; r < ranks.size(); ++r)           \
                    {                                                        \
                        FUNCTION<B>(ranks[r], neighbours[r], f);             \
                    }                                                        \
                });                                                          \
        },                                                                   \
        PERIODIC_ONLY)
        BENCH_PER_RANK("BM_Site_SubdomainExchange", traverse, sites::subdomain_exchange, false);
        BENCH_PER_RANK("BM_Site_MeshSubMesh", build, sites::mesh_sub_mesh, false);
        BENCH_PER_RANK("BM_Site_Interfaces", traverse, sites::interfaces, false);
        BENCH_PER_RANK("BM_Site_PeriodicGhosts", traverse, sites::periodic_ghosts, true);
#undef BENCH_PER_RANK
        site("BM_Site_DiffusionPeel",
             [](auto& ctx)
             {
                 call_sites::PartitionDiffusionPeel<B>(ctx);
             });
    }

    namespace detail
    {
        template <class B, std::size_t dim, class Keep>
        void register_sites(const std::vector<MeshCase>& cases, bool partition, Keep&& keep)
        {
            const auto add = [&](const char* site, bool periodic_only, auto run)
            {
                std::vector<MeshCase> site_cases;
                std::copy_if(cases.begin(),
                             cases.end(),
                             std::back_inserter(site_cases),
                             [&](const MeshCase& c)
                             {
                                 return (!periodic_only || c.geometry == Geometry::periodic) && keep(site, dim, c);
                             });
                if (site_cases.empty())
                {
                    return;
                }
                const std::string name = std::string(B::name) + "/" + site + "<" + std::to_string(dim) + ">";
                mesh_args(benchmark::RegisterBenchmark(name,
                                                       [site = std::string(site), partition, run](benchmark::State& state)
                                                       {
                                                           Timing<B, dim> ctx(state, site, partition);
                                                           run(ctx);
                                                       }),
                          site_cases);
            };
            if (partition)
            {
                for_each_partition_site<B, dim>(add);
            }
            else
            {
                for_each_call_site<B, dim>(add);
            }
        }

        inline bool keep_all(std::string_view, std::size_t, const MeshCase&)
        {
            return true;
        }
    } // namespace detail

    // Registers each site on the meshes c of `cases` for which
    // keep(name, dim, c) holds, all of them by default; the periodic ghosts
    // only on the periodic meshes, the others have none.
    template <class B, std::size_t dim, class Keep = bool (*)(std::string_view, std::size_t, const MeshCase&)>
    void register_call_sites(const std::vector<MeshCase>& cases, Keep&& keep = detail::keep_all)
    {
        detail::register_sites<B, dim>(cases, false, keep);
    }

    // Registers the sites of an MPI rank on the partitions of `cases`
    // (partition_cases), as <B::name>/BM_Site_X<dim>/<args>/<ranks>, with
    // the same filter. The periodic ghosts only on the periodic partitions.
    template <class B, std::size_t dim, class Keep = bool (*)(std::string_view, std::size_t, const MeshCase&)>
    void register_partition_sites(const std::vector<MeshCase>& cases, Keep&& keep = detail::keep_all)
    {
        detail::register_sites<B, dim>(cases, true, keep);
    }

    // The main of the binaries that register sites: fails if a backend
    // computed other sets than the reference.
    inline int run_benchmarks(int argc, char** argv)
    {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv))
        {
            return EXIT_FAILURE;
        }
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        if (const auto failures = verification_failures(); failures > 0)
        {
            std::cerr << failures << " benchmark(s) computed other sets than the reference, see above" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace bench
