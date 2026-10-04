#include <iostream>

#include <samurai/box.hpp>
#include <samurai/samurai.hpp>
#include <samurai/uniform_mesh.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::UniformConfig<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    // 4 is the level of the cells
    samurai::UniformMesh<config_t> mesh(box, 4);

    // Print the number of cells, without and with the ghost cells
    using mesh_id_t     = typename decltype(mesh)::mesh_id_t;
    const auto n_cells  = mesh.nb_cells(mesh_id_t::cells);
    const auto n_ghosts = mesh.nb_cells(mesh_id_t::cells_and_ghosts);
    std::cout << "cells: " << n_cells << "\n";
    std::cout << "cells and ghosts: " << n_ghosts << std::endl;

    samurai::finalize();
    return 0;
}
