// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one: the call sites of bench_set_call_sites with both backends,
// on the main 2D and 3D meshes of bench::mesh_cases, on each geometry, and
// on their partitions into 4 subdomains. Each
// site builds its expressions inside the timed loop, as the library does, so
// the dynamic figure includes the construction of their heap-allocated
// nodes. The ratio dynamic/BM_Site_X / static/BM_Site_X is the price of the
// runtime tree.

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
    // The most fronts at the default max level, stencil radius 1, on the
    // three geometries; with the sweep, the meshes of bench::sweep_cases.
    template <std::size_t dim>
    std::vector<bench::MeshCase> mesh_cases()
    {
        constexpr std::size_t nballs    = (dim == 2) ? 16 : 8;
        constexpr std::size_t max_level = bench::default_max_level<dim>();
        std::vector<bench::MeshCase> cases{
            {max_level, nballs, 1},
            {max_level - 2, nballs, 1, bench::Geometry::hole},
            {max_level, nballs, 1, bench::Geometry::periodic}
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

    // Without the sites the dynamic algebra gets wrong: their checksums
    // differ from the reference (site_checksum.hpp). On 2026-10-19:
    // - the prediction ghosts of the mesh construction, alone
    //   (MeshPredictionGhosts, and its Cached variant) or within
    //   MRMesh::update_sub_mesh_impl (MeshSubMesh, on whole meshes and on
    //   partitions): samurai::dyn::expand of a set projection returns other
    //   cells than the static expansion, takes 290 s per iteration in 3D
    //   against 1 to 2 ms, and fails the assertion n_traversers == 1 of
    //   Projection<DynamicSetAdaptor> in a build with assertions;
    // - the contraction along every axis in 3D (Mode_Contract):
    //   samurai::dyn::contract returns other cells than the static one,
    //   which matches domain_boundary(mesh, level);
    // - a translation of a node that the same expression also reads
    //   untranslated: the subdomain exchange of a partition in 3D
    //   (SubdomainExchange), in outer_subdomain_corner, where the layer band
    //   difference(translate(domain, e_z), translate(domain, 0)) comes out
    //   empty along z (16 cells at level 2 on the 2-rank partition), and the
    //   boundary ghosts of Mode_NestedOn and Mode_FlatOn on the domain with a
    //   hole, difference(domain2, translate(domain2, -direction)), which miss
    //   cells at the min level. With a node per use both are right. The
    //   periodic partition has no outer boundary, so its subdomain exchange
    //   is right.
    bool keep(std::string_view site, std::size_t dim, const bench::MeshCase& c)
    {
        constexpr std::array<std::string_view, 3> excluded{"BM_Site_MeshPredictionGhosts",
                                                           "BM_SiteCached_MeshPredictionGhosts",
                                                           "BM_Site_MeshSubMesh"};
        if (std::find(excluded.begin(), excluded.end(), site) != excluded.end())
        {
            return false;
        }
        if (dim == 3 && site == "BM_Mode_Contract")
        {
            return false;
        }
        if (dim == 3 && site == "BM_Site_SubdomainExchange" && c.geometry != bench::Geometry::periodic)
        {
            return false;
        }
        return !(c.geometry == bench::Geometry::hole && (site == "BM_Mode_NestedOn" || site == "BM_Mode_FlatOn"));
    }

    // The partitions of the same meshes into 4 subdomains, in the box and
    // periodic.
    template <std::size_t dim>
    std::vector<bench::MeshCase> partition_cases()
    {
        std::vector<bench::MeshCase> cases;
        for (const auto& c : bench::partition_cases<dim>())
        {
            if (c.ranks == 4)
            {
                cases.push_back(c);
            }
        }
        return cases;
    }

    template <class B>
    bool register_backend()
    {
        bench::register_call_sites<B, 2>(mesh_cases<2>(), keep);
        bench::register_call_sites<B, 3>(mesh_cases<3>(), keep);
        bench::register_partition_sites<B, 2>(partition_cases<2>(), keep);
        bench::register_partition_sites<B, 3>(partition_cases<3>(), keep);
        return true;
    }

    const bool registered_static  = register_backend<bench::SamuraiStatic>();
    const bool registered_dynamic = register_backend<bench::SamuraiDynamic>();
} // namespace

int main(int argc, char** argv)
{
    return bench::run_benchmarks(argc, argv);
}
