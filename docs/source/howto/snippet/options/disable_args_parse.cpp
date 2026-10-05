#include <iostream>

#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize("Fixed levels", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});

    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(5).disable_args_parse();
    auto mesh = samurai::mra::make_mesh(box, config);

    std::cout << "min level: " << mesh.min_level()
              << ", max level: " << mesh.max_level() << std::endl;

    samurai::finalize();
    return 0;
}
