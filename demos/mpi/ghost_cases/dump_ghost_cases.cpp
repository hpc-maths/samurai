// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Visual validation of the parallel ghost-update robustness suite.
//
// This tool rebuilds, one file per case, EVERY mesh and domain decomposition
// swept by tests/mpi/test_ghost_update_parallel.cpp and writes it to disk as
// HDF5/XDMF so the cases can be inspected in ParaView. The case catalog is
// shared with the test through tests/mpi/ghost_cases.hpp, so what you see here
// is - by construction - exactly what the test validates.
//
// Two suites, mirroring the two test oracles:
//   A_2d_*  the analytic affine oracle matrix (2D, Dirichlet):
//           stencil size x geometry x decomposition.
//   B_2d_*  the decomposition-independence matrix in 2D, and
//   B_3d_*  the same in 3D: geometry x domain shape x stencil x periodicity x
//           decomposition.
//
// Each file carries three fields on the (possibly redistributed) mesh:
//   rank   the MPI rank owning each cell  -> shows the decomposition,
//   level  the refinement level           -> shows the geometry / level jumps,
//   u      the analytic affine field      -> the field the test puts on it.
//
// Run it under MPI so the `rank` field actually reflects a partition:
//   mpiexec -n 4 ./mpi-ghost-cases
//   mpiexec -n 3 ./mpi-ghost-cases --suite B --dim 3 --filter randomhash
//   mpiexec -n 4 ./mpi-ghost-cases --path /tmp/ghost_cases
//
// Sequentially (np = 1) everything lands on rank 0: the geometries and level
// jumps are still meaningful, the decomposition is not.

#include <filesystem>
#include <string>

#include <fmt/format.h>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>

#include "ghost_cases.hpp"

namespace fs  = std::filesystem;
namespace gc  = samurai::ghost_cases;
namespace mpi = boost::mpi;

namespace
{
    struct Options
    {
        fs::path path     = fs::current_path() / "ghost_cases_output";
        std::string suite = "all"; // A | B | all
        int dim           = 0;     // 0 = all, otherwise 2 or 3
        std::string filter;        // keep only cases whose label contains this substring
    };

    // Attach the analytic affine field, then the rank and level diagnostics on
    // the (post-decomposition) mesh, and write the case to <path>/<name>.{h5,xdmf}.
    template <std::size_t Dim, class Mesh>
    void save_case(const Options& opt, Mesh& mesh, bool periodic, const std::string& name)
    {
        mpi::communicator world;

        auto u = samurai::make_scalar_field<double>("u", mesh);
        gc::fill_affine<Dim>(u, periodic);

        auto rank_  = samurai::make_scalar_field<int>("rank", mesh);
        auto level_ = samurai::make_scalar_field<int>("level", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   rank_[cell]  = world.rank();
                                   level_[cell] = static_cast<int>(cell.level);
                               });

        samurai::save(opt.path, fmt::format("{}_np{}", name, world.size()), mesh, rank_, level_, u);
    }

    bool keep(const Options& opt, const std::string& label)
    {
        return opt.filter.empty() || label.find(opt.filter) != std::string::npos;
    }

    // Suite A: the 2D analytic affine oracle matrix (always non-periodic).
    std::size_t dump_suite_A(const Options& opt)
    {
        mpi::communicator world;
        std::size_t n = 0;
        for (const auto& c : gc::make_cases())
        {
            const std::string label = gc::case_label(c);
            if (!keep(opt, label))
            {
                continue;
            }
            auto mesh = gc::build_mesh<2>(c.geom, c.stencil_size, /*periodic=*/false);
            auto u    = samurai::make_scalar_field<double>("u", mesh);
            gc::fill_affine<2>(u, /*periodic=*/false);
            gc::apply_decomposition<2>(c.decomp, u);
            save_case<2>(opt, u.mesh(), /*periodic=*/false, "A_2d_" + label);
            if (world.rank() == 0)
            {
                std::cout << "  A_2d_" << label << std::endl;
            }
            ++n;
        }
        return n;
    }

    // Suite B: the decomposition-independence matrix, in 2D and 3D.
    template <std::size_t Dim>
    std::size_t dump_suite_B(const Options& opt)
    {
        mpi::communicator world;
        std::size_t n = 0;
        for (const auto& c : gc::make_icases())
        {
            const std::string label = gc::icase_label(c);
            if (!keep(opt, label))
            {
                continue;
            }
            gc::DomainCorner<Dim> lo, hi;
            gc::domain_bounds<Dim>(c.domain, lo, hi);
            auto mesh = gc::build_mesh_on_domain<Dim>(c.geom, c.domain, c.stencil_size, c.periodic, lo, hi);
            auto u    = samurai::make_scalar_field<double>("u", mesh);
            gc::fill_affine<Dim>(u, c.periodic);
            gc::apply_decomposition<Dim>(c.decomp, u);
            const std::string name = fmt::format("B_{}d_{}", Dim, label);
            save_case<Dim>(opt, u.mesh(), c.periodic, name);
            if (world.rank() == 0)
            {
                std::cout << "  " << name << std::endl;
            }
            ++n;
        }
        return n;
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Dump the ghost-update robustness cases (meshes + decompositions) for visual validation", argc, argv);

    Options opt;
    app.add_option("--path", opt.path, "Output directory")->capture_default_str();
    app.add_option("--suite", opt.suite, "Which oracle matrix to dump: A | B | all")->capture_default_str();
    app.add_option("--dim", opt.dim, "Restrict suite B to a dimension: 0 (all) | 2 | 3")->capture_default_str();
    app.add_option("--filter", opt.filter, "Keep only cases whose name contains this substring")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    mpi::communicator world;
    if (!fs::exists(opt.path))
    {
        fs::create_directories(opt.path);
    }

    const bool want_A = opt.suite == "A" || opt.suite == "all";
    const bool want_B = opt.suite == "B" || opt.suite == "all";
    const bool want_2 = opt.dim == 0 || opt.dim == 2;
    const bool want_3 = opt.dim == 0 || opt.dim == 3;

    if (world.rank() == 0)
    {
        std::cout << fmt::format("Dumping ghost-update cases to {} (np = {})", opt.path.string(), world.size()) << std::endl;
    }

    std::size_t n = 0;
    if (want_A && want_2)
    {
        n += dump_suite_A(opt);
    }
    if (want_B && want_2)
    {
        n += dump_suite_B<2>(opt);
    }
    if (want_B && want_3)
    {
        n += dump_suite_B<3>(opt);
    }

    if (world.rank() == 0)
    {
        std::cout << fmt::format("Wrote {} case(s) to {}", n, opt.path.string()) << std::endl;
    }

    samurai::finalize();
    return 0;
}
