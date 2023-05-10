#include <boost/mpi.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>

namespace fs = std::filesystem;

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
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

int main()
{
    constexpr std::size_t dim       = 2;
    constexpr std::size_t min_level = 8;
    constexpr std::size_t max_level = 8;

    double a   = 1.;
    double Tf  = 1.;
    double cfl = 0.95 / pow(2, dim - 1);

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

    output << mesh << std::endl;
    output << "-----------------------------------" << std::endl;
    output << mesh.domain() << std::endl;
    output << "-----------------------------------" << std::endl;

    auto u = init(mesh);
    samurai::make_bc<samurai::Dirichlet>(u, 0.);
    // output << u << std::endl;

    auto unp1 = samurai::make_field<double, size>("unp1", mesh);
    auto rank = samurai::make_field<double, size>("rank", mesh);
    rank.fill(world.rank());

    double dt      = a * cfl * samurai::cell_length(max_level);
    double t       = 0.;
    std::size_t nt = 0;

    while (t != Tf)
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

        samurai::update_ghost_subdomains(u);
        samurai::update_ghost_mr(u);
        unp1.resize();
        unp1.fill(0);

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
        samurai::save(fmt::format("advection_1d_ite_{}", nt), mesh, u, rank);
        nt++;
    }
    output << u << std::endl;

    // {
    //     std::array<mpi::request, 2> reqs;
    //     samurai::Box<double, 1> box = {{1}, {2}};
    //     using Config                = samurai::MRConfig<dim>;
    //     samurai::MRMesh<Config> mesh_0, mesh_1{box, 2, 4};
    //     reqs[0] = world.isend(0, 1, mesh_1);
    //     reqs[1] = world.irecv(0, 0, mesh_0);
    //     mpi::wait_all(reqs.begin(), reqs.end());
    //     output << world.rank() << mesh_0 << ", " << std::endl;
    // }

    return 0;
}
