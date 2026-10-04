#include <algorithm>
#include <cmath>
#include <iostream>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

// Largest value of |f - g| over the cells of all ranks
template <class Field>
double max_diff(const Field& f, const Field& g)
{
    double result = 0;
    auto update   = [&](const auto& cell)
    {
        result = std::max(result, std::abs(f[cell] - g[cell]));
    };
    samurai::for_each_cell(f.mesh(), update);
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    result = mpi::all_reduce(world, result, mpi::maximum<double>());
#endif
    return result;
}

// Non-linear diffusion: the flux u * grad(u) and its Jacobian,
// as in demos/FiniteVolume/heat_nonlinear.cpp
template <class Field>
auto make_nonlinear_diffusion()
{
    static constexpr std::size_t dim = Field::dim;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, Field, Field>;

    samurai::FluxDefinition<cfg> flux_definition;

    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            flux_definition[d].cons_flux_function =
                [](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u)
            {
                auto u_mean = (u[0] + u[1]) / 2;
                auto grad_u = (u[0] - u[1]) / data.cell_length;
                flux        = u_mean * grad_u;
            };

            flux_definition[d].cons_jacobian_function =
                [](samurai::StencilJacobian<cfg>& jac, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u)
            {
                auto u_mean = (u[0] + u[1]) / 2;
                auto grad_u = (u[0] - u[1]) / data.cell_length;
                jac[0]      = grad_u / 2 + u_mean / data.cell_length;
                jac[1]      = grad_u / 2 - u_mean / data.cell_length;
            };
        });

    auto scheme = samurai::make_flux_based_scheme(flux_definition);
    scheme.set_name("nonlinear_diffusion");
    return scheme;
}

void solve_nonlinear_heat()
{
    samurai::Box<double, 2> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<2>().min_level(5).max_level(5);
    auto mesh   = samurai::mra::make_mesh(box, config);

    auto bump = [](const auto& x)
    {
        double r2 = std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2);
        return 1 + std::exp(-100 * r2);
    };
    auto u    = samurai::make_scalar_field<double>("u", mesh, bump);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh, 0.);
    // unp1 = u copies the boundary conditions of u: give u the same
    samurai::make_bc<samurai::Neumann<1>>(u, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

    double dt = 1e-3;
    auto diff = make_nonlinear_diffusion<decltype(u)>();
    auto id   = samurai::make_identity<decltype(u)>();
    auto A    = id + dt * diff;

    auto solver = samurai::petsc::make_solver(A);
    solver.set_unknown(unp1);
    solver.configure = [](SNES& snes, KSP& ksp, PC& pc)
    {
        SNESSetType(snes, SNESNEWTONLS); // -snes_type newtonls
        KSPSetType(ksp, KSPPREONLY);     // -ksp_type preonly
        PCSetType(pc, PCLU);             // -pc_type lu
    };
    solver.stop_program_on_divergence(false);

    for (int n = 1; n <= 3; ++n)
    {
        unp1 = u; // initial guess of the Newton method

        auto reason = solver.solve(u); // solves A(unp1) = u
        if (reason < 0)
        {
            std::cout << "Newton diverged at step " << n << std::endl;
            break;
        }

        auto Au  = A(unp1);
        double r = max_diff(u, Au);
        int its  = solver.iterations();
        std::cout << "step " << n << ": " << its << " iterations";
        std::cout << ", residual " << r << std::endl;

        samurai::swap(u, unp1);
    } // end of the time loop
}

int main(int argc, char** argv)
{
    samurai::initialize("Implicit non-linear heat", argc, argv);
    SAMURAI_PARSE(argc, argv);

    solve_nonlinear_heat(); // its solver is destroyed on return

    samurai::finalize();
    return 0;
}
