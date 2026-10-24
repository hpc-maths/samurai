// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The benchmarks of the call sites (set_call_sites.hpp), for one backend of
// the set algebra (set_algebra_backend.hpp). register_call_sites<B, dim>
// registers them as <B::name>/BM_Site_X<dim>, so that
// --benchmark_filter='^<name>/' selects the backend.
//
// Args: (max_level, number of balls, stencil radius), see bench::mesh_cases.
//
// Each site is written once against a context, Timing, which gives its
// inputs and runs it in a benchmark.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_backend.hpp"
#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
#include "site_inputs.hpp"

namespace bench
{
    // The context of a site in a benchmark of backend B: its inputs come
    // from the files, and it runs in the timed loop.
    template <class B, std::size_t dim>
    class Timing
    {
      public:

        explicit Timing(benchmark::State& state)
            : m_state(state)
        {
        }

        const backend_inputs_t<B, dim>& inputs(bool moved = false)
        {
            return backend_inputs<B, dim>(m_state, moved);
        }

        // site(f) passes each set it evaluates to f(level, set), which
        // returns its number of intervals. The callback only counts: the
        // time is that of the set algebra in the library function, to set
        // against BM_SiteOp_X or BM_Op_* of bench_set_library_ops.
        template <class Site>
        void traverse(Site&& site)
        {
            m_state.counters["sets"] = static_cast<double>(run_site<B>(m_state, site));
        }

      private:

        benchmark::State& m_state;
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
    } // namespace call_sites

    // Calls v(name, run) for each site, where run(ctx) runs it in the context
    // ctx (Timing of B in dim).
    template <class B, std::size_t dim, class V>
    void for_each_call_site(V&& v)
    {
        // A site that only reads the inputs of the mesh.
#define BENCH_SIMPLE(NAME, FUNCTION)     \
    v(NAME,                              \
      [](auto& ctx)                      \
      {                                  \
          const auto& in = ctx.inputs(); \
          ctx.traverse(                  \
              [&](auto&& f)              \
              {                          \
                  FUNCTION<B>(in, f);    \
              });                        \
      })
#define BENCH_SITE(NAME, FUNCTION) \
    v(NAME,                        \
      [](auto& ctx)                \
      {                            \
          FUNCTION<B>(ctx);        \
      })
        BENCH_SITE("BM_Site_MeshPredictionGhosts", call_sites::MeshPredictionGhosts);
        BENCH_SITE("BM_SiteCached_MeshPredictionGhosts", call_sites::CachedMeshPredictionGhosts);
        BENCH_SIMPLE("BM_Site_Corners", sites::corners);
        BENCH_SIMPLE("BM_Site_GhostProjection", sites::ghost_projection);
        BENCH_SIMPLE("BM_Site_GhostPrediction", sites::ghost_prediction);
        BENCH_SITE("BM_Site_OuterGhosts", call_sites::OuterGhosts);
        BENCH_SITE("BM_SiteCached_OuterGhosts", call_sites::CachedOuterGhosts);
        BENCH_SIMPLE("BM_Site_Detail", sites::detail);
        BENCH_SIMPLE("BM_Site_Tag", sites::tag);
        BENCH_SITE("BM_Site_FieldTransfer", call_sites::FieldTransfer);
        BENCH_SIMPLE("BM_Site_Interfaces", sites::interfaces);
#undef BENCH_SIMPLE
#undef BENCH_SITE
    }

    namespace detail
    {
        inline bool keep_all(std::string_view, std::size_t, const MeshCase&)
        {
            return true;
        }
    } // namespace detail

    // Registers each site on the meshes c of `cases` for which
    // keep(name, dim, c) holds, all of them by default.
    template <class B, std::size_t dim, class Keep = bool (*)(std::string_view, std::size_t, const MeshCase&)>
    void register_call_sites(const std::vector<MeshCase>& cases, Keep&& keep = detail::keep_all)
    {
        for_each_call_site<B, dim>(
            [&](const char* site, auto run)
            {
                std::vector<MeshCase> site_cases;
                std::copy_if(cases.begin(),
                             cases.end(),
                             std::back_inserter(site_cases),
                             [&](const MeshCase& c)
                             {
                                 return keep(site, dim, c);
                             });
                if (site_cases.empty())
                {
                    return;
                }
                const std::string name = std::string(B::name) + "/" + site + "<" + std::to_string(dim) + ">";
                mesh_args(benchmark::RegisterBenchmark(name,
                                                       [run](benchmark::State& state)
                                                       {
                                                           Timing<B, dim> ctx(state);
                                                           run(ctx);
                                                       }),
                          site_cases);
            });
    }
} // namespace bench
