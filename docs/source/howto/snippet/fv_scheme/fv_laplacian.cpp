#include <algorithm>
#include <cmath>
#include <numbers>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using samurai::SchemeType;

static constexpr std::size_t dim = 2;
static constexpr double pi       = std::numbers::pi;

// Laplacian as the divergence of the normal gradient
template <class Field>
auto make_my_laplacian()
{
    using cfg = samurai::FluxConfig<SchemeType::LinearHomogeneous,
                                    2,      // stencil size
                                    Field,  // output field
                                    Field>; // input field

    // (u_R - u_L) / h, the same in every direction
    samurai::FluxDefinition<cfg> normal_grad(
        [](samurai::FluxStencilCoeffs<cfg>& c, double h)
        {
            c[0] = -1 / h; // left cell of the stencil
            c[1] = 1 / h;  // right cell of the stencil
        });

    auto lap = samurai::make_flux_based_scheme(normal_grad);
    lap.set_name("my Laplacian");
    return lap;
}

// Largest |f - g| over the cells
template <class F, class G>
double max_diff(const F& f, const G& g)
{
    double r = 0;
    samurai::for_each_cell(f.mesh(),
                           [&](const auto& cell)
                           {
                               r = std::max(r, std::abs(f[cell] - g[cell]));
                           });
    return r;
}

int main(int argc, char** argv)
{
    samurai::initialize("User-defined Laplacian", argc, argv);
    SAMURAI_PARSE(argc, argv);

    auto f = [](const auto& x)
    {
        return std::sin(pi * x[0]) * std::sin(pi * x[1]);
    };
    auto lap_f = [&](const auto& x)
    {
        return -2 * pi * pi * f(x);
    };

    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    fmt::print("{:>5}  {:>9}  {:>9}  {:>5}\n", "level", "built-in", "error", "order");
    double previous_error = 0;
    for (std::size_t level = 4; level <= 8; ++level)
    {
        auto config = samurai::mesh_config<dim>();
        config.min_level(level).max_level(level);
        auto mesh = samurai::mra::make_mesh(box, config);

        auto u = samurai::make_scalar_field<double>("u", mesh, f);
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto exact = samurai::make_scalar_field<double>("exact", mesh, lap_f);

        using field_t = decltype(u);
        auto my_lap   = make_my_laplacian<field_t>();
        auto built_in = samurai::make_laplacian_order2<field_t>();

        auto lap_u = my_lap(u);

        double diff  = max_diff(lap_u, built_in(u));
        double error = max_diff(lap_u, exact);
        fmt::print("{:5}  {:9.2e}  {:9.2e}", level, diff, error);
        if (previous_error > 0)
        {
            fmt::print("  {:5.2f}", std::log2(previous_error / error));
        }
        fmt::print("\n");
        previous_error = error;
    }

    samurai::finalize();
    return 0;
}
