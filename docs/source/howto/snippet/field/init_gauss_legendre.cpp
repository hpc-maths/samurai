#include <iostream>

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
    config.min_level(0).max_level(1);
    auto mesh = samurai::mra::make_mesh(box, config);

    auto f = [](const auto& x)
    {
        return x[0] * x[0];
    };
    // Exact for polynomials up to degree 2
    samurai::GaussLegendre<2> gl;

    auto u = samurai::make_scalar_field<double>("u", mesh, f, gl);

    std::cout << u << std::endl;

    samurai::finalize();
    return 0;
}
