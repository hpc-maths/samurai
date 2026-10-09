#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "print_ghosts.hpp"

int main(int argc, char** argv)
{
    samurai::initialize("Dirichlet and Neumann", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    // 4 x 4 cells on the unit square, u = 1 in every cell
    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(2);
    auto mesh = samurai::mra::make_mesh(box, config);
    auto u    = samurai::make_scalar_field<double>("u", mesh, 1.);

    const samurai::DirectionVector<dim> left   = {-1, 0};
    const samurai::DirectionVector<dim> right  = {1, 0};
    const samurai::DirectionVector<dim> bottom = {0, -1};
    const samurai::DirectionVector<dim> top    = {0, 1};

    // u = 0 on the left side
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.)->on(left);

    // u = y on the right side
    auto u_right =
        [](const auto& /* direction */, const auto& /* cell_in */, const auto& x)
    {
        return x[1];
    };
    samurai::make_bc<samurai::Dirichlet<1>>(u, u_right)->on(right);

    // du/dn = 0 on the bottom and top sides
    samurai::make_bc<samurai::Neumann<1>>(u, 0.)->on(bottom, top);

    // A vector field: one value per component
    auto v = samurai::make_vector_field<double, 2>("v", mesh, 1.);
    samurai::make_bc<samurai::Dirichlet<1>>(v, 0., 2.);

    // Fill the ghosts, then read them
    samurai::update_ghost_mr(u, v);

    print_ghosts("u, left", u, left);
    print_ghosts("u, right", u, right);
    print_ghosts("u, bottom", u, bottom);
    print_ghosts("u, top", u, top);
    print_ghosts("v, left", v, left);
    print_ghosts("v, right", v, right);
    print_ghosts("v, bottom", v, bottom);
    print_ghosts("v, top", v, top);

    samurai::finalize();
    return 0;
}
