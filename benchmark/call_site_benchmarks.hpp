// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The benchmarks of the call sites (set_call_sites.hpp) and of the ways to
// consume a set (set_modes.hpp), for one backend of the set algebra
// (set_algebra_backend.hpp). register_call_sites<B, dim> registers them as
// <B::name>/BM_Site_X<dim> and <B::name>/BM_Mode_X<dim>, so that
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
#include "set_modes.hpp"
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

        // site(emit) builds level cell arrays and passes each one it keeps
        // to emit(level, lca): the time is that of building them, `itv` and
        // `cells` count what they keep.
        template <class Site>
        void build(Site&& site)
        {
            m_state.counters["sets"] = static_cast<double>(run_build_site<B>(m_state, site));
        }

        // site(f) passes each set it evaluates to f(level, set), which only
        // asks whether it is empty.
        template <class Site>
        void empty(Site&& site)
        {
            m_state.counters["sets"] = static_cast<double>(run_empty_site<B>(m_state, site));
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
    } // namespace call_sites

    // Calls v(name, run) for each site and each mode, where run(ctx) runs it
    // in the context ctx (Timing of B in dim).
    template <class B, std::size_t dim, class V>
    void for_each_call_site(V&& v)
    {
        // A site or mode that only reads the inputs of the mesh: KIND is
        // traverse, build or empty (see Timing).
#define BENCH_SIMPLE(NAME, KIND, FUNCTION) \
    v(NAME,                                \
      [](auto& ctx)                        \
      {                                    \
          const auto& in = ctx.inputs();   \
          ctx.KIND(                        \
              [&](auto&& f)                \
              {                            \
                  FUNCTION<B>(in, f);      \
              });                          \
      })
#define BENCH_SITE(NAME, FUNCTION) \
    v(NAME,                        \
      [](auto& ctx)                \
      {                            \
          FUNCTION<B>(ctx);        \
      })
        BENCH_SITE("BM_Site_MeshPredictionGhosts", call_sites::MeshPredictionGhosts);
        BENCH_SITE("BM_SiteCached_MeshPredictionGhosts", call_sites::CachedMeshPredictionGhosts);
        BENCH_SIMPLE("BM_Site_Corners", traverse, sites::corners);
        BENCH_SIMPLE("BM_Site_GhostProjection", traverse, sites::ghost_projection);
        BENCH_SIMPLE("BM_Site_GhostPrediction", traverse, sites::ghost_prediction);
        BENCH_SITE("BM_Site_OuterGhosts", call_sites::OuterGhosts);
        BENCH_SITE("BM_SiteCached_OuterGhosts", call_sites::CachedOuterGhosts);
        BENCH_SIMPLE("BM_Site_Detail", traverse, sites::detail);
        BENCH_SIMPLE("BM_Site_Tag", traverse, sites::tag);
        BENCH_SITE("BM_Site_FieldTransfer", call_sites::FieldTransfer);
        BENCH_SIMPLE("BM_Site_Interfaces", traverse, sites::interfaces);
        BENCH_SIMPLE("BM_Site_MeshSubMesh", build, sites::mesh_sub_mesh);
        BENCH_SIMPLE("BM_Site_ConstructUnion", build, sites::construct_union);
        BENCH_SIMPLE("BM_Site_BuildPyramid", build, sites::build_pyramid);
        BENCH_SIMPLE("BM_Site_CellArrayFromTag", build, sites::cell_array_from_tag);
        BENCH_SIMPLE("BM_Site_AmrSubMesh", build, sites::amr_sub_mesh);
        BENCH_SITE("BM_Site_BcCells", call_sites::BcCells);
        BENCH_SIMPLE("BM_Site_BcCorners", traverse, sites::bc_corners);
        BENCH_SIMPLE("BM_Site_BcFurtherGhosts", traverse, sites::bc_further_ghosts);
        BENCH_SIMPLE("BM_Site_PerCellChildren", traverse, sites::per_cell_children);
        BENCH_SITE("BM_Site_OwnershipPerCell", call_sites::OwnershipPerCell);

        BENCH_SIMPLE("BM_Mode_Traverse", traverse, modes::traverse_proj_cells);
        BENCH_SIMPLE("BM_Mode_Materialize", build, modes::materialize_proj_cells);
        BENCH_SIMPLE("BM_Mode_Build", build, modes::build_proj_cells);
        BENCH_SIMPLE("BM_Mode_EmptyHit", empty, modes::empty_hit);
        BENCH_SIMPLE("BM_Mode_EmptyMiss", empty, modes::empty_miss);
        BENCH_SIMPLE("BM_Mode_Contract", traverse, modes::contract);
        BENCH_SIMPLE("BM_Mode_ContractMasked", traverse, modes::contract_masked);
        BENCH_SIMPLE("BM_Mode_ExpandUniform", traverse, modes::expand_uniform);
        BENCH_SIMPLE("BM_Mode_ExpandAxes", traverse, modes::expand_axes);
        BENCH_SIMPLE("BM_Mode_ExpandMasked", traverse, modes::expand_masked);
        BENCH_SIMPLE("BM_Mode_OnMixedLevels", traverse, modes::on_mixed_levels);
        BENCH_SIMPLE("BM_Mode_OnSameLevel", traverse, modes::on_same_level);
        BENCH_SIMPLE("BM_Mode_NestedOn", traverse, modes::nested_on);
        BENCH_SIMPLE("BM_Mode_FlatOn", traverse, modes::flat_on);
        BENCH_SIMPLE("BM_Mode_IntersectionSelfOn", traverse, modes::intersection_self_on);
        BENCH_SIMPLE("BM_Mode_SelfOn", traverse, modes::self_on);
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
