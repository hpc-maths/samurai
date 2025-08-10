// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <filesystem>
#include <random>
#include <cmath>
namespace fs = std::filesystem;

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& uv, const std::string& suffix = "")
{
    auto mesh   = uv.mesh();
    auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, uv, level_);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, uv, level_);
    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, uv);
#endif
}

/**
 * Gray–Scott reaction–diffusion system (explicit FV scheme):
 *  U_t = D_u * ΔU - U V^2 + F (1 - U)
 *  V_t = D_v * ΔV + U V^2 - (F + k) V
 */
template <std::size_t dim>
int main_dim(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the Gray-Scott system (explicit)", argc, argv);

    using Config  = samurai::MRConfig<dim, 3>;
    using Box     = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;

    std::cout << "------------------------- Gray-Scott -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Domain
    double left_box  = -1.0;
    double right_box = 1.0;

    // Physical parameters
    double Du = 2e-5;  // diffusion of U
    double Dv = 1e-5;  // diffusion of V
    double F  = 0.04;  // feed
    double k  = 0.06;  // kill

    // Initial condition
    std::string init_sol = "spot"; // "spot" or "random"
    double spot_radius   = 0.1 * (right_box - left_box);
    double U_in          = 0.50;
    double V_in          = 0.25;

    // Time integration
    double Tf  = 5.0;
    double dt  = 0.0;     // if 0, computed from CFL on diffusion
    double cfl = 0.95;
    double t   = 0.0;
    std::string restart_file;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = dim == 1 ? 9 : 7;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "grayscott_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 100;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Du", Du, "Diffusion coefficient for U")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Dv", Dv, "Diffusion coefficient for V")->capture_default_str()->group("Simulation parameters");
    app.add_option("--F", F, "Feed rate")->capture_default_str()->group("Simulation parameters");
    app.add_option("--k", k, "Kill rate")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: spot/random")->capture_default_str()->group("Simulation parameters");
    app.add_option("--spot-radius", spot_radius, "Radius for the spot initial condition (absolute units)")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--U-in", U_in, "Initial U value inside the spot")->capture_default_str()->group("Simulation parameters");
    app.add_option("--V-in", V_in, "Initial V value inside the spot")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step (0 = computed from CFL)")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL (diffusion)")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files (0 = only last)")->capture_default_str()->group("Output");

    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh;

    static constexpr std::size_t n_comp = 2;
    auto uv                            = samurai::make_vector_field<double, n_comp>("uv", mesh);

    if (restart_file.empty())
    {
        mesh = {box, min_level, max_level};
        uv.resize();

        if (init_sol == "spot")
        {
            // Standard GS: U=1, V=0 everywhere, one spot in the center perturbed
            samurai::for_each_cell(mesh,
                                   [&](auto& cell)
                                   {
                                       double r2 = 0.0;
                                       for (std::size_t d = 0; d < dim; ++d)
                                       {
                                           double x = cell.center(d);
                                           r2 += x * x;
                                       }
                                       double r = std::sqrt(r2);
                                       uv[cell][0] = (r <= spot_radius) ? U_in : 1.0;
                                       uv[cell][1] = (r <= spot_radius) ? V_in : 0.0;
                                   });
        }
        else if (init_sol == "random")
        {
            std::mt19937 gen(42);
            std::uniform_real_distribution<double> noise(-0.05, 0.05);
            samurai::for_each_cell(mesh,
                                   [&](auto& cell)
                                   {
                                       uv[cell][0] = 1.0 + noise(gen);
                                       uv[cell][1] = 0.0 + noise(gen);
                                   });
        }
        else
        {
            std::cerr << "Unmanaged initial solution '" << init_sol << "'.";
            return EXIT_FAILURE;
        }
    }
    else
    {
        samurai::load(restart_file, mesh, uv);
    }

    // Boundary conditions (zero-flux Neumann)
    samurai::make_bc<samurai::Neumann<1>>(uv);

    // Intermediary fields for the RK3 scheme
    auto uv1   = samurai::make_vector_field<double, n_comp>("uv1", mesh);
    auto uv2   = samurai::make_vector_field<double, n_comp>("uv2", mesh);
    auto unp1  = samurai::make_vector_field<double, n_comp>("unp1", mesh);
    uv1.copy_bc_from(uv);
    uv2.copy_bc_from(uv);
    unp1.copy_bc_from(uv);

    // Diffusion operator (component-wise coefficients)
    samurai::DiffCoeff<n_comp> Kc; // one coefficient per component
    Kc(0) = Du;
    Kc(1) = Dv;
    auto diff = samurai::make_multi_diffusion_order2<decltype(uv)>(Kc);

    // Reaction operator (local, nonlinear)
    using cfg  = samurai::LocalCellSchemeConfig<samurai::SchemeType::NonLinear, decltype(uv), decltype(uv)>;
    auto react = samurai::make_cell_based_scheme<cfg>();
    react.set_name("Reaction");
    react.set_scheme_function(
        [F, k](const auto& cell, const auto& field) -> samurai::SchemeValue<cfg>
        {
            auto w = field[cell];
            double U = w[0];
            double V = w[1];
            samurai::SchemeValue<cfg> rhs;
            rhs[0] = -U * V * V + F * (1.0 - U);
            rhs[1] = U * V * V - (F + k) * V;
            return rhs;
        });

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx   = mesh.cell_length(max_level);
        double Dmax = std::max(Du, Dv);
        dt          = cfl * (dx * dx) / (std::pow(2.0, static_cast<int>(dim)) * Dmax);
    }

    auto MRadaptation = samurai::make_MRAdapt(uv);
    auto mra_config   = samurai::mra_config();
    MRadaptation(mra_config);

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    if (nfiles != 1)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, uv, suffix);
    }

    auto rhs = [&](auto& f)
    {
        return react(f) - diff(f);
    };

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

        // Mesh adaptation
        MRadaptation(mra_config);
        uv1.resize();
        uv2.resize();
        unp1.resize();

        // TVD-RK3 (SSPRK3) on rhs = react(uv) - diff(uv)
        uv1   = uv + dt * rhs(uv);
        uv2   = 3.0 / 4.0 * uv + 1.0 / 4.0 * (uv1 + dt * rhs(uv1));
        unp1  = 1.0 / 3.0 * uv + 2.0 / 3.0 * (uv2 + dt * rhs(uv2));

        // uv <-- unp1
        samurai::swap(uv, unp1);

        // Save the result
        if (nfiles == 0 || t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            if (nfiles != 1)
            {
                std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                save(path, filename, uv, suffix);
            }
            else
            {
                save(path, filename, uv);
            }
        }

        std::cout << std::endl;
    }

    if constexpr (dim == 1)
    {
        std::cout << std::endl;
        std::cout << "Run the following command to view the results:" << std::endl;
        std::cout << "python <<path to samurai>>/python/read_mesh.py " << filename
                  << "_ite_ --field uv level --start 0 --end " << nsave << std::endl;
    }

    samurai::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    static constexpr std::size_t dim = 2;
    return main_dim<dim>(argc, argv);
}


