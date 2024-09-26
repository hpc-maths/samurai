// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

template <std::size_t dim>
double exact_solution(xt::xtensor_fixed<double, xt::xshape<dim>> coords, double t)
{
    const double c = 1; // constant parameter

    double result = 1;
    for (std::size_t d = 0; d < dim; ++d)
    {
        result *= coords(d) * coords(d) / (c - 6 * t);
    }
    return result;
}

template <class Field>
auto make_nonlinear_diffusion()
{
    static constexpr std::size_t dim               = Field::dim;
    static constexpr std::size_t field_size        = Field::size;
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    samurai::FluxDefinition<cfg> flux;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto integral_constant_d)
        {
            static constexpr std::size_t d = integral_constant_d();

            flux[d].cons_flux_function = [](auto& cells, const Field& u)
            {
                auto& L = cells[0];
                auto& R = cells[1];
                auto dx = L.length;

                auto _u     = (u[L] + u[R]) / 2;
                auto grad_u = (u[L] - u[R]) / dx;

                samurai::FluxValue<cfg> f = _u * grad_u; // (1)
                return f;
            };

            flux[d].cons_jacobian_function = [](auto& cells, const Field& u)
            {
                auto& L = cells[0];
                auto& R = cells[1];
                auto dx = L.length;

                samurai::StencilJacobian<cfg> jac;
                auto& jac_L = jac[0];
                auto& jac_R = jac[1];

                auto _u     = (u[L] + u[R]) / 2;
                auto grad_u = (u[L] - u[R]) / dx;

                jac_L = grad_u / 2 + _u / dx; // derive (1) w.r.t. u[L]
                jac_R = grad_u / 2 - _u / dx; // derive (1) w.r.t. u[R]

                return jac;
            };
        });

    return samurai::make_flux_based_scheme(flux);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Non-linear heat -------------------------" << std::endl;

    /*
        Solves the non-linear heat equation
                ∂u/∂t + ∇・(u∇u) = 0,
        with exact solution
                u(x,t) = x²/(c-6t), where c is a constant.
        This is 3.2. Example 2 in
        Exact solutions of nonlinear diffusion equations by variational iteration method, A. Sadighi, D.D. Ganji, 2007
        https://www.sciencedirect.com/science/article/pii/S0898122107002957#b22
    */

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = 0;
    double right_box = 1;

    // Time integration
    double Tf            = 1.;
    double dt            = 1e-4;
    bool explicit_scheme = false;
    double cfl           = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 4;
    std::size_t max_level = 4;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path              = fs::current_path();
    std::string filename       = "heat_nonlinear_" + std::to_string(dim) + "D";
    bool save_final_state_only = false;

    CLI::App app{"Finite volume example for the heat equation in 2D"};
    app.add_flag("--explicit", explicit_scheme, "Explicit scheme instead of implicit")->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_flag("--save-final-state-only", save_final_state_only, "Save final state only")->group("Output");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    //------------------//
    // Petsc initialize //
    //------------------//

    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off");

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u = samurai::make_field<1>("u",
                                    mesh,
                                    [&](const auto& coords)
                                    {
                                        return exact_solution(coords, 0);
                                    });

    auto unp1 = samurai::make_field<1>("unp1", mesh);

    samurai::make_bc<samurai::Dirichlet<1>>(u,
                                            [&](const auto&, const auto&, const auto& coords)
                                            {
                                                return exact_solution(coords, 0);
                                            });

    auto diff = make_nonlinear_diffusion<decltype(u)>();
    auto id   = samurai::make_identity<decltype(u)>();

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (explicit_scheme)
    {
        double diff_coeff = 1;
        double dx         = mesh.cell_length(max_level);
        dt                = cfl * (dx * dx) / (pow(2, dim) * diff_coeff);
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    std::size_t nsave = 0, nt = 0;
    if (!save_final_state_only)
    {
        save(path, filename, u, fmt::format("_ite_{}", nsave++));
    }

    double t = 0;
    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt) << std::flush;

        // Update boundary conditions
        if (explicit_scheme)
        {
            u.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                    [&](const auto&, const auto&, const auto& coords)
                                                    {
                                                        return exact_solution(coords, t - dt);
                                                    });
        }
        else
        {
            unp1.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet<1>>(unp1,
                                                    [&](const auto&, const auto&, const auto& coords)
                                                    {
                                                        return exact_solution(coords, t);
                                                    });
        }

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        unp1.resize();

        if (explicit_scheme)
        {
            unp1 = u - dt * diff(u);
        }
        else
        {
            samurai::petsc::solve(id + dt * diff, unp1, u); // solves the non-linear equation [id+dt*diff](unp1) = u
        }

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        double error = samurai::L2_error(u,
                                         [&](const auto& coords)
                                         {
                                             return exact_solution(coords, t);
                                         });
        std::cout.precision(2);
        std::cout << ", L2-error: " << std::scientific << error;

        // Save the result
        if (!save_final_state_only)
        {
            save(path, filename, u, fmt::format("_ite_{}", nsave++));
        }

        std::cout << std::endl;
    }

    if (save_final_state_only)
    {
        save(path, filename, u);
    }

    PetscFinalize();
    samurai::finalize();
    return 0;
}
