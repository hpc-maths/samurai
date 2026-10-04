#include <cmath>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>

int main(int argc, char** argv)
{
    static constexpr std::size_t dim = 2;

    samurai::initialize("Scoped timer example", argc, argv);

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(5);
    auto mesh = samurai::mra::make_mesh(box, config);

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto init = [&](const auto& cell)
    {
        const double x = cell.center(0) - 0.5;
        const double y = cell.center(1) - 0.5;
        u[cell]        = std::exp(-20. * (x * x + y * y));
    };

    {
        samurai::ScopedTimer init_timer("init field");
        samurai::for_each_cell(mesh, init);
    } // init_timer stops here

    samurai::finalize();
    return 0;
}
