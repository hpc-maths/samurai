#include <boost/mpi.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>

namespace fs = std::filesystem;

#include <CLI/CLI.hpp>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/memory.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>

namespace mpi = boost::mpi;

static constexpr std::size_t size = 1;

template <class Mesh>
auto init(Mesh& mesh)
{
    auto u = samurai::make_field<double, size>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto center         = cell.center();
                               const double radius = .2;

                               const double x_center = -0.5;
                               if constexpr (Mesh::dim == 2)
                               {
                                   if (std::abs(center[0] - x_center) <= radius && std::abs(center[1] - x_center) <= radius)
                                   {
                                       u[cell] = 1;
                                   }
                               }
                               else
                               {
                                   if (std::abs(center[0] - x_center) <= radius)
                                   {
                                       u[cell] = 1;
                                   }
                               }
                           });

    return u;
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    std::size_t min_level     = 4;
    std::size_t max_level     = 6;

    double a             = 1.;
    double Tf            = 1.;
    double cfl           = 0.45;
    double mr_epsilon    = 1e-4; // Threshold used by multiresolution
    double mr_regularity = 1.;   // Regularity guess for multiresolution

    std::size_t nite = 0;
    CLI::App app{};
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--nite", nite, "number of iterations");

    CLI11_PARSE(app, argc, argv);

    mpi::environment env;
    mpi::communicator world;

    auto output_name = fmt::format("output_{}.log", world.rank());

    auto output = std::ofstream(output_name);

    samurai::Box<double, dim> box;
    if constexpr (dim == 1)
    {
        box = samurai::Box<double, dim>{{-1}, {1}};
    }
    else if constexpr (dim == 2)
    {
        box = samurai::Box<double, dim>{
            {-1, -1},
            {1,  1 }
        };
    }
    else if constexpr (dim == 3)
    {
        box = samurai::Box<double, dim>{
            {-1, -1, -1},
            {1,  1,  1 }
        };
    }

    using Config = samurai::MRConfig<dim>;
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u = init(mesh);
    samurai::make_bc<samurai::Dirichlet>(u, 0.);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);
    // MRadaptation(mr_epsilon, mr_regularity);

    auto rank = samurai::make_field<double, size>("rank", mesh);
    rank.fill(world.rank());
    samurai::save(fmt::format("advection_{}d_{}_init", dim, world.size()), mesh, u, rank);

    // samurai::check_duplicate_cells(u);
    // return 0;

    auto unp1 = samurai::make_field<double, size>("unp1", mesh);
    rank.resize();
    rank.fill(world.rank());

    double dt      = a * cfl * samurai::cell_length(max_level);
    double t       = 0.;
    std::size_t nt = 0;

    // samurai::save(std::filesystem::current_path(), fmt::format("advection_{}d_{}_adapt", dim, world.size()), {true, true}, mesh, u);
    // samurai::save(fmt::format("advection_{}d_{}_adapt", dim, world.size()), mesh, u, rank);

    // while (t != Tf)
    for (std::size_t ite = 0; ite < nite; ++ite)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        if (world.rank() == 0)
        {
            std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt, t, dt) << std::endl;
        }

        MRadaptation(mr_epsilon, mr_regularity);

        samurai::update_ghost_mr(u);

        samurai::check_duplicate_cells(u);
        unp1.resize();
        unp1.fill(0);
        rank.resize();
        rank.fill(world.rank());

        if constexpr (dim == 1)
        {
            unp1 = u - dt * samurai::upwind(a, u);
        }
        else
        {
            std::array<double, dim> a_;
            a_.fill(a);
            unp1 = u - dt * samurai::upwind(a_, u);
        }

        std::swap(u.array(), unp1.array());
        samurai::save(fmt::format("advection_{}d_{}_ite_{}", dim, world.size(), nt), mesh, rank, u);

        nt++;
    }

    return 0;
}
