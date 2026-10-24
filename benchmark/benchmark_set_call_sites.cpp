// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Time the set algebra takes in the operations samurai runs at every time
// step, on the sets of MR-adapted meshes in 1D, 2D and 3D. Needs only the
// interval structures and the set algebra: the sets are read from the files
// of generate_set_algebra_inputs. bench_set_library_ops times the same
// operations with the library machinery.
//
// The benchmarks are named <backend>/BM_Site_X<dim>/<args>, see
// call_site_benchmarks.hpp. To compare another set algebra, register its
// backend below (SET_ALGEBRA.md, "Adding a backend").

#include <benchmark/benchmark.h>

#include "call_site_benchmarks.hpp"
#include "set_algebra_backend.hpp"
#include "site_inputs.hpp"

namespace
{
    template <class B>
    bool register_backend()
    {
        bench::register_call_sites<B, 1>(bench::mesh_cases<1>());
        bench::register_call_sites<B, 2>(bench::mesh_cases<2>());
        bench::register_call_sites<B, 3>(bench::mesh_cases<3>());
        bench::register_partition_sites<B, 1>(bench::partition_cases<1>());
        bench::register_partition_sites<B, 2>(bench::partition_cases<2>());
        bench::register_partition_sites<B, 3>(bench::partition_cases<3>());
        return true;
    }

    const bool registered = register_backend<bench::SamuraiStatic>();
} // namespace

int main(int argc, char** argv)
{
    return bench::run_benchmarks(argc, argv);
}
