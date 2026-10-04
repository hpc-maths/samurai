#include <iostream>

#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});

    // Prediction stencil radius of 2
    auto config = samurai::mesh_config<dim, 2>();
    config.min_level(2).max_level(5);
    // The scheme reads 3 cells in each direction
    config.max_stencil_radius(3);
    // Levels differ by at most one within 3 cells
    config.graduation_width(3);
    auto mesh = samurai::mra::make_mesh(box, config);

    // The same mesh with the default settings
    auto default_config = samurai::mesh_config<dim>();
    default_config.min_level(2).max_level(5);
    auto default_mesh = samurai::mra::make_mesh(box, default_config);

    std::cout << "ghost width: " << mesh.ghost_width() << "\n";
    const auto default_width = default_mesh.ghost_width();
    std::cout << "default ghost width: " << default_width << std::endl;

    samurai::finalize();
    return 0;
}
