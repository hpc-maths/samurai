#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "print_ghosts.hpp"

// Robin condition u + du/dn = g on the boundary face
template <class Field>
struct RobinImpl : public samurai::Bc<Field>
{
    INIT_BC(RobinImpl, 2)

    // The boundary cell and the ghost next to it,
    // written for the right boundary
    stencil_t get_stencil(constant_stencil_size_t) const final
    {
        return samurai::line_stencil<dim, 0>(0, 1);
    }

    apply_function_t
    get_apply_function(constant_stencil_size_t, const direction_t&) const final
    {
        return [](Field& u, const stencil_cells_t& c, const value_t& g)
        {
            const auto& in  = c[0];
            const auto& out = c[1];
            const double h  = u.mesh().cell_length(out.level);

            // (u_in + u_out) / 2 + (u_out - u_in) / h = g
            u[out] = (2 * h * g - (h - 2) * u[in]) / (h + 2);
        };
    }
};

struct Robin
{
    template <class Field>
    using impl_t = RobinImpl<Field>;
};

int main(int argc, char** argv)
{
    samurai::initialize("User-defined condition", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    // 4 x 4 cells on the unit square
    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(2);
    auto mesh = samurai::mra::make_mesh(box, config);

    // u = x at the cell centers
    auto u = samurai::make_scalar_field<double>("u",
                                                mesh,
                                                [](const auto& x)
                                                {
                                                    return x[0];
                                                });

    // g = u + du/dn for u = x: du/dn is the x-component of n
    auto g = [](const auto& n, const auto&, const auto& x)
    {
        return x[0] + n[0];
    };
    samurai::make_bc<Robin>(u, g);

    samurai::update_ghost_mr(u);

    print_ghosts("left", u, {-1, 0});
    print_ghosts("right", u, {1, 0});
    print_ghosts("bottom", u, {0, -1});

    samurai::finalize();
    return 0;
}
