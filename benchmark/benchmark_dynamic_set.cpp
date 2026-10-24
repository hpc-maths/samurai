// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one: the call sites of bench_set_call_sites with both backends,
// on the main 2D and 3D meshes of bench::mesh_cases and, with the sweep, on
// those of bench::sweep_cases. Each site builds its expressions inside the
// timed loop, as the library does, so the dynamic figure includes the
// construction of their heap-allocated nodes. The ratio dynamic/BM_Site_X /
// static/BM_Site_X is the price of the runtime tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "call_site_benchmarks.hpp"
#include "set_algebra_backend.hpp"
#include "set_algebra_dynamic_backend.hpp"
#include "site_inputs.hpp"

namespace
{
    // The most fronts at the default max level, stencil radius 1; with the
    // sweep, the meshes of bench::sweep_cases.
    template <std::size_t dim>
    std::vector<bench::MeshCase> mesh_cases()
    {
        constexpr std::size_t nballs    = (dim == 2) ? 16 : 8;
        constexpr std::size_t max_level = bench::default_max_level<dim>();
        std::vector<bench::MeshCase> cases{
            {max_level, nballs, 1}
        };
        if constexpr (bench::sweep_enabled)
        {
            for (const auto& c : bench::sweep_cases<dim>())
            {
                if (c.max_level != max_level)
                {
                    cases.push_back(c);
                }
            }
        }
        return cases;
    }

    // Without the sites the dynamic algebra gets wrong: they compute other
    // sets than the static algebra. On 2026-10-19:
    // - the prediction ghosts of the mesh construction, alone
    //   (MeshPredictionGhosts, and its Cached variant) or within
    //   MRMesh::update_sub_mesh_impl (MeshSubMesh): samurai::dyn::expand of a
    //   set projection returns other cells than the static expansion, takes
    //   290 s per iteration in 3D against 1 to 2 ms, and fails the assertion
    //   n_traversers == 1 of Projection<DynamicSetAdaptor> in a build with
    //   assertions;
    // - the contraction along every axis in 3D (Mode_Contract):
    //   samurai::dyn::contract returns other cells than the static one,
    //   which matches domain_boundary(mesh, level).
    bool keep(std::string_view site, std::size_t dim, const bench::MeshCase&)
    {
        constexpr std::array<std::string_view, 3> excluded{"BM_Site_MeshPredictionGhosts",
                                                           "BM_SiteCached_MeshPredictionGhosts",
                                                           "BM_Site_MeshSubMesh"};
        if (std::find(excluded.begin(), excluded.end(), site) != excluded.end())
        {
            return false;
        }
        return !(dim == 3 && site == "BM_Mode_Contract");
    }

    template <class B>
    bool register_backend()
    {
        bench::register_call_sites<B, 2>(mesh_cases<2>(), keep);
        bench::register_call_sites<B, 3>(mesh_cases<3>(), keep);
        return true;
    }

    const bool registered_static  = register_backend<bench::SamuraiStatic>();
    const bool registered_dynamic = register_backend<bench::SamuraiDynamic>();
} // namespace

BENCHMARK_MAIN();
