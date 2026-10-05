#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
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

    auto u = samurai::make_scalar_field<double>("u", mesh);

    auto set_value = [&](const auto& cell)
    {
        u[cell] = cell.center(0);
    };
    samurai::for_each_cell<samurai::Run::Parallel>(mesh, set_value);

    samurai::finalize();
    return 0;
}
