#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

#include "print_ghosts.hpp"

int main(int argc, char** argv)
{
    samurai::initialize("Boundary regions", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    // 4 x 4 cells on the unit square, u = w = 0 in every cell
    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(2);
    auto mesh = samurai::mra::make_mesh(box, config);
    auto u    = samurai::make_scalar_field<double>("u", mesh, 0.);
    auto w    = samurai::make_scalar_field<double>("w", mesh, 0.);

    const samurai::DirectionVector<dim> left   = {-1, 0};
    const samurai::DirectionVector<dim> right  = {1, 0};
    const samurai::DirectionVector<dim> bottom = {0, -1};
    const samurai::DirectionVector<dim> top    = {0, 1};

    // u = 0 on the whole boundary, then u = 1 where y > 0.5
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    auto upper = samurai::make_bc_region(mesh,
                                         [](const auto& x)
                                         {
                                             return x[1] > 0.5;
                                         });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 1.)->on(upper);

    // w = 1 on the boundary of the cells of the left column
    auto left_column = samurai::difference(
        mesh.domain(),
        samurai::translate(mesh.domain(), -left));
    samurai::make_bc<samurai::Dirichlet<1>>(w, 1.)->on(left_column);

    samurai::update_ghost_mr(u, w);

    print_ghosts("u, left", u, left);
    print_ghosts("w, left", w, left);
    print_ghosts("w, right", w, right);
    print_ghosts("w, bottom", w, bottom);
    print_ghosts("w, top", w, top);

    samurai::finalize();
    return 0;
}
