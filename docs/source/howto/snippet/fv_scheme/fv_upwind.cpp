#include <algorithm>
#include <cmath>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using samurai::SchemeType;

static constexpr std::size_t dim = 2;

// Upwind scheme for div(a u), with a velocity field a
template <class Field, class VelocityField>
auto make_my_upwind(VelocityField& a)
{
    using cfg = samurai::FluxConfig<SchemeType::LinearHeterogeneous,
                                    2,
                                    Field,
                                    Field,
                                    VelocityField>; // parameter field

    samurai::FluxDefinition<cfg> upwind;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            upwind[d].cons_flux_function =
                [&a](samurai::FluxStencilCoeffs<cfg>& c,
                     const samurai::StencilData<cfg>& data)
            {
                const auto& cells = data.cells;
                // Velocity on the face: average of the two cells
                double a_d = (a[cells[0]](d) + a[cells[1]](d)) / 2;
                c[0]       = a_d >= 0 ? a_d : 0; // left cell
                c[1]       = a_d >= 0 ? 0 : a_d; // right cell
            };
        });

    auto scheme = samurai::make_flux_based_scheme(upwind);
    scheme.set_parameter_field(a); // keeps the ghosts of a up to date
    return scheme;
}

int main(int argc, char** argv)
{
    samurai::initialize("User-defined upwind scheme", argc, argv);
    SAMURAI_PARSE(argc, argv);

    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(6).max_level(6);
    auto mesh = samurai::mra::make_mesh(box, config);

    // Rotation around the center of the square
    auto a = samurai::make_vector_field<double, dim>("a", mesh, 0.);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               auto x     = cell.center();
                               a[cell](0) = 0.5 - x[1];
                               a[cell](1) = x[0] - 0.5;
                           });

    auto u = samurai::make_scalar_field<double>(
        "u",
        mesh,
        [](const auto& x)
        {
            double r2 = std::pow(x[0] - 0.7, 2) + std::pow(x[1] - 0.5, 2);
            return std::exp(-100 * r2);
        });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    using field_t = decltype(u);
    auto my_conv  = make_my_upwind<field_t>(a);
    auto built_in = samurai::make_convection_upwind<field_t>(a);

    auto v   = my_conv(u);
    auto w   = built_in(u);
    double r = 0;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               r = std::max(r, std::abs(v[cell] - w[cell]));
                           });
    fmt::print("difference with make_convection_upwind: {:.2e}\n", r);

    samurai::finalize();
    return 0;
}
