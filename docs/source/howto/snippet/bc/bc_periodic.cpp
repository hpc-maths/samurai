#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "print_ghosts.hpp"

int main(int argc, char** argv)
{
    samurai::initialize("Periodic direction", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    // Periodic in x, not in y
    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(2).periodic({true, false});
    auto mesh = samurai::mra::make_mesh(box, config);

    // u = x at the cell centers
    auto u = samurai::make_scalar_field<double>("u",
                                                mesh,
                                                [](const auto& x)
                                                {
                                                    return x[0];
                                                });

    // Applied only in y, the direction that is not periodic
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    samurai::update_ghost_mr(u);

    print_ghosts("left", u, {-1, 0});
    print_ghosts("right", u, {1, 0});
    print_ghosts("bottom", u, {0, -1});

    samurai::finalize();
    return 0;
}
