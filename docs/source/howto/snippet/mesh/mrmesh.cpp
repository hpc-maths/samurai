#include <iostream>

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

    // Print the number of cells of each level
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    for (auto l = mesh.min_level(); l <= mesh.max_level(); ++l)
    {
        const auto n = mesh.nb_cells(l, mesh_id_t::cells);
        std::cout << "level " << l << ": " << n << " cells\n";
    }

    samurai::finalize();
    return 0;
}
