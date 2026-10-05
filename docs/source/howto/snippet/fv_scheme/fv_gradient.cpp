#include <algorithm>
#include <cmath>
#include <numbers>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using samurai::SchemeType;

static constexpr std::size_t dim = 2;
static constexpr double pi       = std::numbers::pi;

// Gradient of a scalar field: the output field has dim components
template <class Field>
auto make_my_gradient()
{
    using mesh_t   = typename Field::mesh_t;
    using value_t  = typename Field::value_type;
    using output_t = samurai::VectorField<mesh_t, value_t, dim>;

    using cfg = samurai::FluxConfig<SchemeType::LinearHomogeneous, 2, output_t, Field>;

    samurai::FluxDefinition<cfg> average;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            // (u_L + u_R) / 2 in component d, 0 in the others:
            // c[i] is a dim x 1 matrix
            average[d].cons_flux_function =
                [](samurai::FluxStencilCoeffs<cfg>& c, double /* h */)
            {
                c[0].fill(0);
                c[1].fill(0);
                c[0](d, 0) = 0.5;
                c[1](d, 0) = 0.5;
            };
        });

    return samurai::make_flux_based_scheme(average);
}

// Largest |f_d - g_d| over the cells and the components
template <class F, class G>
double max_diff(const F& f, const G& g)
{
    double r = 0;
    samurai::for_each_cell(f.mesh(),
                           [&](const auto& cell)
                           {
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   double e = f[cell](d) - g[cell](d);
                                   r        = std::max(r, std::abs(e));
                               }
                           });
    return r;
}

int main(int argc, char** argv)
{
    samurai::initialize("User-defined gradient", argc, argv);
    SAMURAI_PARSE(argc, argv);

    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>();
    config.min_level(6).max_level(6);
    auto mesh = samurai::mra::make_mesh(box, config);

    auto u = samurai::make_scalar_field<double>("u",
                                                mesh,
                                                [](const auto& x)
                                                {
                                                    return std::sin(pi * x[0])
                                                         * std::sin(pi * x[1]);
                                                });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    auto my_grad  = make_my_gradient<decltype(u)>();
    auto built_in = samurai::make_gradient_order2<decltype(u)>();

    fmt::print("difference with make_gradient_order2: {:.2e}\n",
               max_diff(my_grad(u), built_in(u)));

    samurai::finalize();
    return 0;
}
