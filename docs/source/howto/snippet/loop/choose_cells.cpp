#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(5);
    auto mesh = samurai::mra::make_mesh(box, config);

    using mesh_id_t      = typename decltype(mesh)::mesh_id_t;
    const auto max_level = mesh.max_level();
    auto print_cell      = [](const auto& cell)
    {
        std::cout << cell.level << " " << cell.center() << "\n";
    };
    auto print_row = [](auto level, const auto& i, const auto& index)
    {
        std::cout << level << " " << i << " " << index << "\n";
    };

    // The cells and their ghost cells
    auto& with_ghosts = mesh[mesh_id_t::cells_and_ghosts];
    samurai::for_each_cell(with_ghosts, print_cell);

    // The cells of one level
    auto& cells = mesh[mesh_id_t::cells][max_level];
    samurai::for_each_interval(cells, print_row);

    // The ghost cells of one level only, built with the set algebra
    auto& all   = mesh[mesh_id_t::cells_and_ghosts][max_level];
    auto ghosts = samurai::difference(all, cells);
    samurai::for_each_cell(mesh, ghosts, print_cell);

    samurai::finalize();
    return 0;
}
