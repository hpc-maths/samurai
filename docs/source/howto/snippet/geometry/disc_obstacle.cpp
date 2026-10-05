#include <cmath>
#include <iostream>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    // Level-set function of a disc of radius 0.2 centered at
    // (0.5, 0.5): negative in the obstacle, positive in the fluid
    auto phi = [](const auto& x)
    {
        double dx = x[0] - 0.5;
        double dy = x[1] - 0.5;
        return std::sqrt(dx * dx + dy * dy) - 0.2;
    };

    // 1. Mesh the bounding box: every cell is at the maximum level
    samurai::Box<double, dim> box({0.0, 0.0}, {2.0, 1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(6);
    auto mesh = samurai::mra::make_mesh(box, config);

    // 2. Keep the cells whose center is in the fluid
    using mesh_t  = decltype(mesh);
    using cl_type = typename mesh_t::cl_type;
    cl_type cl(mesh.origin_point(), mesh.scaling_factor());
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               if (phi(cell.center()) > 0)
                               {
                                   cl[cell.level].add_cell(cell);
                               }
                           });

    // 3. Build the mesh from those cells
    mesh = samurai::mra::make_mesh(cl, config);

    using mesh_id_t = typename mesh_t::mesh_id_t;
    std::size_t n   = mesh.nb_cells(mesh_id_t::cells);
    double dx       = mesh.cell_length(mesh.max_level());
    double area     = static_cast<double>(n) * dx * dx;
    double exact    = 2.0 - M_PI * 0.2 * 0.2;
    std::cout << "cells: " << n << '\n';
    std::cout << "covered area: " << area << '\n';
    std::cout << "exact area: " << exact << '\n';

    // 4. Adapt the mesh to a field that varies near the obstacle
    auto u_init = [&](const auto& x)
    {
        return std::exp(-50 * phi(x));
    };
    auto u = samurai::make_scalar_field<double>("u", mesh, u_init);
    samurai::make_bc<samurai::Neumann<1>>(u, 0.);

    auto adapt = samurai::make_MRAdapt(u);
    adapt(samurai::mra_config().epsilon(1e-3));

    auto& cells = mesh[mesh_id_t::cells];
    for (std::size_t l = mesh.min_level(); l <= mesh.max_level(); ++l)
    {
        std::size_t nl = cells[l].nb_cells();
        std::cout << "level " << l << ": " << nl << '\n';
    }

    auto lvl = samurai::make_scalar_field<std::size_t>("level", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               lvl[cell] = cell.level;
                           });
    samurai::save("disc_obstacle", mesh, u, lvl);

    samurai::finalize();
    return 0;
}
