#include <cmath>
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

    auto init = [&](auto level, const auto& i, const auto& index)
    {
        const double h = mesh.cell_length(level);
        const auto o   = mesh.origin_point();
        const auto j   = index[0];
        // Cell centers, relative to the center (0.5, 0.5) of the box
        auto x = h * (xt::arange(i.start, i.end) + 0.5) + o[0] - 0.5;
        auto y = h * (j + 0.5) + o[1] - 0.5;

        u(level, i, j) = xt::exp(-20. * (x * x + y * y));
    };
    samurai::for_each_interval(mesh, init);

    samurai::finalize();
    return 0;
}
