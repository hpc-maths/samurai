// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// Generates the small reference samurai files consumed by test_yt_reader.py.
// The field `u` is set to the analytic oracle
//
//     u(center) = sum_d center[d] * 10^d
//
// so that the yt reconstruction can verify value <-> geometry <-> ordering
// without any tolerance ambiguity.  Regenerate with (see tools/yt/README.md):
//
//   <mpicxx> -DSAMURAI_WITH_MPI ... generate_reference.cpp -o generate_reference
//   ./generate_reference                    # writes ref_1d/2d/3d.h5
//   mpirun -n 2 ./generate_reference --mpi  # writes ref_2d_mpi.h5
//
// The output directory is the first CLI argument (default: current directory).
#include <cmath>
#include <string>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

template <class Coord>
double analytic(const Coord& c, std::size_t dim)
{
    double v = 0., w = 1.;
    for (std::size_t d = 0; d < dim; ++d)
    {
        v += w * c[d];
        w *= 10.;
    }
    return v;
}

template <std::size_t dim>
auto make_graded_mesh_and_field(std::size_t max_level, double eps, double sharpness)
{
    using Box = samurai::Box<double, dim>;
    typename Box::point_t lo, hi;
    lo.fill(0.);
    hi.fill(1.);
    Box box(lo, hi);

    auto config = samurai::mesh_config<dim>().min_level(2).max_level(max_level).disable_minimal_ghost_width();
    auto mesh   = samurai::mra::make_mesh(box, config);
    auto u      = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto cell)
                           {
                               auto c    = cell.center();
                               double r2 = 0.;
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   r2 += (c[d] - 0.5) * (c[d] - 0.5);
                               }
                               u[cell] = std::exp(-sharpness * r2);
                           });

    auto adapt = samurai::make_MRAdapt(u);
    adapt(samurai::mra_config().epsilon(eps));

    samurai::for_each_cell(mesh,
                           [&](auto cell)
                           {
                               u[cell] = analytic(cell.center(), dim);
                           });
    return std::make_pair(std::move(mesh), std::move(u));
}

// A uniform mesh at a single level: every cell is a leaf at the coarsest level,
// exercising the root grid (real values, no refinement) on the reader side.
template <std::size_t dim>
auto make_uniform_mesh_and_field(std::size_t level)
{
    using Box = samurai::Box<double, dim>;
    typename Box::point_t lo, hi;
    lo.fill(0.);
    hi.fill(1.);
    Box box(lo, hi);

    auto config = samurai::mesh_config<dim>().min_level(level).max_level(level).disable_minimal_ghost_width();
    auto mesh   = samurai::mra::make_mesh(box, config);
    auto u      = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto cell)
                           {
                               u[cell] = analytic(cell.center(), dim);
                           });
    return std::make_pair(std::move(mesh), std::move(u));
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    std::string dir = ".";
    bool mpi        = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--mpi")
        {
            mpi = true;
        }
        else if (a[0] != '-')
        {
            dir = a;
        }
    }
    auto path = [&](const std::string& f)
    {
        return dir + "/" + f;
    };

    if (mpi)
    {
        auto [mesh, u] = make_graded_mesh_and_field<2>(5, 1e-3, 200.);
        samurai::dump(path("ref_2d_mpi"), mesh, u);
    }
    else
    {
        {
            auto [mesh, u] = make_graded_mesh_and_field<1>(5, 1e-3, 200.);
            samurai::dump(path("ref_1d"), mesh, u);
        }
        {
            auto [mesh, u] = make_graded_mesh_and_field<2>(4, 1e-2, 100.);
            samurai::dump(path("ref_2d"), mesh, u);
        }
        {
            auto [mesh, u] = make_graded_mesh_and_field<3>(3, 1e-2, 30.);
            samurai::dump(path("ref_3d"), mesh, u);
        }
        {
            auto [mesh, u] = make_uniform_mesh_and_field<2>(2);
            samurai::dump(path("ref_uniform"), mesh, u);
        }
    }

    samurai::finalize();
    return 0;
}
