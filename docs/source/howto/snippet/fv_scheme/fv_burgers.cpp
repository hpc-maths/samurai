#include <algorithm>
#include <cmath>
#include <numbers>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using samurai::SchemeType;

static constexpr std::size_t dim = 2;
static constexpr double pi       = std::numbers::pi;

using config_t = samurai::mesh_config<dim>;
using mesh_t   = samurai::MRMesh<config_t>;
using field_t  = samurai::ScalarField<mesh_t, double>;

using cfg = samurai::FluxConfig<SchemeType::NonLinear,
                                2,        // stencil size
                                field_t,  // output field
                                field_t>; // input field

// Flux of div(f(u)) with f(u) = u^2 / 2: (f(u_L) + f(u_R)) / 2
void burgers_flux(samurai::FluxValue<cfg>& flux,
                  const samurai::StencilData<cfg>& /* data */,
                  const samurai::StencilValues<cfg>& u)
{
    flux = (u[0] * u[0] + u[1] * u[1]) / 4;
}

// Derivatives of the flux w.r.t. u_L and u_R
void burgers_jacobian(samurai::StencilJacobian<cfg>& jac,
                      const samurai::StencilData<cfg>& /* data */,
                      const samurai::StencilValues<cfg>& u)
{
    jac[0] = u[0] / 2;
    jac[1] = u[1] / 2;
}

// The same flux, written as a pair of fluxes
void burgers_flux_pair(samurai::FluxValuePair<cfg>& flux,
                       const samurai::StencilData<cfg>& /* data */,
                       const samurai::StencilValues<cfg>& u)
{
    flux[0] = (u[0] * u[0] + u[1] * u[1]) / 4; // into V_L
    flux[1] = -flux[0];                        // into V_R
}

// Largest gap between the Jacobian and finite differences of the flux
double check_jacobian()
{
    samurai::StencilCells<cfg> cells;
    samurai::StencilData<cfg> data(cells);
    samurai::StencilValues<cfg> u = {0.3, -0.7};
    samurai::StencilJacobian<cfg> jac;
    burgers_jacobian(jac, data, u);

    double r       = 0;
    const double e = 1e-6;
    for (std::size_t i = 0; i < 2; ++i)
    {
        samurai::FluxValue<cfg> f_plus;
        samurai::FluxValue<cfg> f_minus;
        auto u_plus  = u;
        auto u_minus = u;
        u_plus[i] += e;
        u_minus[i] -= e;
        burgers_flux(f_plus, data, u_plus);
        burgers_flux(f_minus, data, u_minus);
        double fd = (f_plus - f_minus) / (2 * e);
        r         = std::max(r, std::abs(fd - jac[i]));
    }
    return r;
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
    samurai::initialize("User-defined Burgers flux", argc, argv);
    SAMURAI_PARSE(argc, argv);

    samurai::FluxDefinition<cfg> centered;
    samurai::FluxDefinition<cfg> centered_pair;
    for (std::size_t d = 0; d < dim; ++d)
    {
        centered[d].cons_flux_function     = burgers_flux;
        centered[d].cons_jacobian_function = burgers_jacobian;
        centered_pair[d].flux_function     = burgers_flux_pair;
    }

    // u = sin(2 pi x) sin(2 pi y), div(f(u)) = u (du/dx + du/dy)
    auto u0 = [](const auto& x)
    {
        return std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]);
    };
    auto div_f = [&](const auto& x)
    {
        double dx = std::cos(2 * pi * x[0]) * std::sin(2 * pi * x[1]);
        double dy = std::sin(2 * pi * x[0]) * std::cos(2 * pi * x[1]);
        return u0(x) * 2 * pi * (dx + dy);
    };

    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    fmt::print("{:>5}  {:>9}  {:>9}  {:>5}\n", "level", "pair", "error", "order");
    double previous_error = 0;
    for (std::size_t level = 4; level <= 8; ++level)
    {
        auto config = config_t();
        config.min_level(level).max_level(level).periodic(true);
        auto mesh = samurai::mra::make_mesh(box, config);

        auto u     = samurai::make_scalar_field<double>("u", mesh, u0);
        auto exact = samurai::make_scalar_field<double>("exact", mesh, div_f);

        auto burgers = samurai::make_flux_based_scheme(centered);
        auto pair    = samurai::make_flux_based_scheme(centered_pair);
        burgers.set_name("my Burgers");

        auto div_u = burgers(u);

        double diff  = max_diff(div_u, pair(u));
        double error = max_diff(div_u, exact);
        fmt::print("{:5}  {:9.2e}  {:9.2e}", level, diff, error);
        if (previous_error > 0)
        {
            fmt::print("  {:5.2f}", std::log2(previous_error / error));
        }
        fmt::print("\n");
        previous_error = error;
    }

    fmt::print("Jacobian vs finite differences: {:.2e}\n", check_jacobian());

    samurai::finalize();
    return 0;
}
