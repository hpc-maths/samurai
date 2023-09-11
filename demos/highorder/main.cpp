// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/reconstruction.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field,
          // scheme config
          std::size_t neighbourhood_width = 2,
          class cfg                       = samurai::StarStencilFV<Field::dim, Field::size, neighbourhood_width>,
          class bdry_cfg                  = samurai::BoundaryConfigFV<neighbourhood_width>>
class HighOrderDiffusion : public samurai::CellBasedScheme<HighOrderDiffusion<Field>, cfg, bdry_cfg, Field>
{
    using base_class = samurai::CellBasedScheme<HighOrderDiffusion<Field>, cfg, bdry_cfg, Field>;

  public:

    static constexpr std::size_t dim = Field::dim;
    using field_t                    = Field;
    using directional_bdry_config_t  = typename base_class::directional_bdry_config_t;

    HighOrderDiffusion()
    {
    }

    static constexpr auto stencil()
    {
        return samurai::star_stencil<dim, 2>();
    }

    static std::array<double, 9> coefficients(double h)
    {
        // https://en.wikipedia.org/wiki/Finite_difference_coefficient
        std::array<double, 9> coeffs = {1. / 12, -4. / 3, 5., -4. / 3, 1. / 12, 1. / 12, -4. / 3, -4. / 3, 1. / 12};
        double one_over_h2           = 1 / (h * h);
        for (double& coeff : coeffs)
        {
            coeff *= one_over_h2;
        }
        return coeffs;
    }

    directional_bdry_config_t dirichlet_config(const samurai::DirectionVector<dim>& direction) const override
    {
        directional_bdry_config_t config;

        config.directional_stencil = this->get_directional_stencil(direction);

        static constexpr std::size_t cell1  = 0;
        static constexpr std::size_t cell2  = 1;
        static constexpr std::size_t cell3  = 2;
        static constexpr std::size_t ghost1 = 3;
        static constexpr std::size_t ghost2 = 4;

        // We need to define a polynomial of degree 3 that passes by the 4 points c3, c2, c1 and the boundary point.
        // This polynomial writes
        //                       p(x) = a*x^3 + b*x^2 + c*x + d.
        // The coefficients a, b, c, d are found by inverting the Vandermonde matrix obtained by inserting the 4 points into
        // the polynomial. If we set the abscissa 0 at the center of c1, this system reads
        //                       p(  0) = u1
        //                       p( -h) = u2
        //                       p(-2h) = u3
        //                       p(h/2) = dirichlet_value.
        // Then, we want that the ghost values be also located on this polynomial, i.e.
        //                       u_g1 = p( h)
        //                       u_g2 = p(2h).
        // This gives
        //             5/16 * u_g1  +  15/16 * u1  -5/16 * u2  +  1/16 * u3 = dirichlet_value
        //             5/64 * u_g2  +  45/32 * u1  -5/8  * u2  +  9/64 * u3 = dirichlet_value

        // Equation of ghost1
        config.equations[0].ghost_index        = ghost1;
        config.equations[0].get_stencil_coeffs = [&](double)
        {
            std::array<double, 5> coeffs;
            coeffs[ghost1] = 5. / 16;
            coeffs[cell1]  = 15. / 16;
            coeffs[cell2]  = -5. / 16;
            coeffs[cell3]  = 1. / 16;
            coeffs[ghost2] = 0;
            return coeffs;
        };
        config.equations[0].get_rhs_coeffs = [&](double)
        {
            return 1.;
        };

        // Equation of ghost2
        config.equations[1].ghost_index        = ghost2;
        config.equations[1].get_stencil_coeffs = [&](double)
        {
            std::array<double, 5> coeffs;
            coeffs[ghost2] = 5. / 64;
            coeffs[cell1]  = 45. / 32;
            coeffs[cell2]  = -5. / 8;
            coeffs[cell3]  = 9. / 64;
            coeffs[ghost1] = 0;
            return coeffs;
        };
        config.equations[1].get_rhs_coeffs = [&](double)
        {
            return 1.;
        };

        return config;
    }
};

template <class Field>
auto make_high_order_diffusion()
{
    return HighOrderDiffusion<Field>();
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim              = 2;
    constexpr std::size_t stencil_width    = 2;
    constexpr std::size_t graduation_width = 4;
    constexpr std::size_t prediction_order = 4;
    using Config                           = samurai::MRConfig<dim, stencil_width, graduation_width, prediction_order>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

    // Multiresolution parameters
    std::size_t min_level  = 4;
    std::size_t max_level  = 4;
    std::size_t refinement = 5;
    double mr_epsilon      = 2.e-4; // Threshold used by multiresolution
    double mr_regularity   = 1.;    // Regularity guess for multiresolution
    bool correction        = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "poisson_highorder_2d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example for the advection equation in 2d "
                 "using multiresolution"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", min_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--refinement", refinement, "Number of refinement")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    samurai::Box<double, dim> box(min_corner, max_corner);
    using mesh_t    = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type   = typename mesh_t::cl_type;
    mesh_t init_mesh{box, min_level, max_level};

    PetscInitialize(&argc, &argv, 0, nullptr);
    PetscOptionsSetValue(NULL, "-options_left", "off");

    auto adapt_field = samurai::make_field<double, 1>("adapt_field",
                                                      init_mesh,
                                                      [](const auto& coord)
                                                      {
                                                          const auto& x = coord[0];
                                                          const auto& y = coord[1];
                                                          double radius = 0.4;
                                                          if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < radius * radius)
                                                          {
                                                              return 0;
                                                          }
                                                          else
                                                          {
                                                              return 1;
                                                          }
                                                      });

    auto MRadaptation = samurai::make_MRAdapt(adapt_field);
    MRadaptation(mr_epsilon, mr_regularity);

    // samurai::save("initial_mesh", mesh);
    double h_coarse            = -1;
    double error_coarse        = -1;
    double error_recons_coarse = -1;

    for (std::size_t ite = 0; ite < refinement; ++ite)
    {
        auto mesh = init_mesh;
        for (std::size_t i_ref = 0; i_ref < ite; ++i_ref)
        {
            cl_type cl;
            samurai::for_each_interval(mesh[mesh_id_t::cells],
                                       [&](std::size_t level, const auto& i, const auto& index)
                                       {
                                           samurai::static_nested_loop<dim - 1, 0, 2>(
                                               [&](auto& stencil)
                                               {
                                                   auto new_index = 2 * index + stencil;
                                                   cl[level + 1][new_index].add_interval(i << 1);
                                               });
                                       });
            // mesh = {cl, mesh.min_level() + 1, mesh.max_level() + 1};
            mesh = {cl, min_level + i_ref + 1, max_level + i_ref + 1};
        }
        // std::cout << mesh << std::endl;
        // samurai::save("refine_mesh", mesh);

        // Equation: -Lap u = f   in [0, 1]^2
        //            f(x,y) = 2(y(1-y) + x(1-x))
        auto f = samurai::make_field<double, 1>("f",
                                                mesh,
                                                [](const auto& coord)
                                                {
                                                    const auto& x = coord[0];
                                                    const auto& y = coord[1];
                                                    // return 2 * (y*(1 - y) + x * (1 - x));
                                                    // return 2 * pow(4 * M_PI, 2) * sin(4 * M_PI * x)*sin(4 * M_PI *
                                                    // y);
                                                    return (-pow(y, 4) - 2 * x * (1 + 2 * x * y * y)) * exp(x * y * y);
                                                });

        // samurai::for_each_cell(mesh[mesh_id_t::reference], [&](auto& cell)
        // {
        //     double x = cell.center(0);
        //     double y = cell.center(1);
        //     f[cell] =  2 * (y*(1 - y) + x * (1 - x));
        // });

        // std::size_t level = mesh.max_level();
        // auto set = samurai::intersection(mesh[mesh_id_t::cells][level],
        // mesh[mesh_id_t::reference][level-1]).on(level-1);
        // set.apply_op(samurai::projection(f));
        // auto f_recons = samurai::make_field<double, 1>("f_recons", mesh);
        // auto error_f = samurai::make_field<double, 1>("error", mesh);
        // set.apply_op(samurai::prediction<prediction_order, true>(f_recons, f));
        // samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level,
        // const auto& i, const auto& index)
        // {
        //     auto j = index[0];
        //     error_f(level, i, j) = xt::abs(f(level, i, j) - f_recons(level, i,
        //     j));
        // });
        // samurai::save("test_pred", mesh, f, f_recons, error_f);
        // return 0;

        auto u = samurai::make_field<double, 1>("u", mesh);
        samurai::make_bc<samurai::Dirichlet>(u,
                                             [](const auto&, const auto& coord)
                                             {
                                                 const auto& x = coord[0];
                                                 const auto& y = coord[1];
                                                 // return 0.;
                                                 return exp(x * y * y);
                                             });
        u.fill(0);

        HighOrderDiffusion<decltype(u)> diff;

        auto solver = samurai::petsc::make_solver(diff);
        KSP ksp     = solver.Ksp();
        PC pc;
        KSPGetPC(ksp, &pc);
        KSPSetType(ksp, KSPPREONLY); // (equiv. '-ksp_type preonly')
        PCSetType(pc, PCLU);         // (equiv. '-pc_type lu')
        solver.solve(u, f);

        auto exact_func = [](const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            // return x * (1 - x) * y*(1 - y);
            // return sin(4 * M_PI * x)*sin(4 * M_PI *
            // y);
            return exp(x * y * y);
        };

        double h = samurai::cell_length(mesh.min_level());

        double error = L2_error(u, exact_func);
        std::cout.precision(2);
        std::cout << "refinement: " << ite << std::endl;
        std::cout << "L2-error         : " << std::scientific << error;
        if (h_coarse != -1)
        {
            double convergence_order = samurai::convergence_order(h, error, h_coarse, error_coarse);
            std::cout << " (order = " << std::defaultfloat << convergence_order << ")";
        }
        std::cout << std::endl;

        samurai::update_ghost_mr(u);
        auto u_recons = samurai::reconstruction(u);

        double error_recons = L2_error(u_recons, exact_func);

        std::cout.precision(2);
        std::cout << "L2-error (recons): " << std::scientific << error_recons;

        if (h_coarse != -1)
        {
            double convergence_order = samurai::convergence_order(h, error_recons, h_coarse, error_recons_coarse);
            std::cout << " (order = " << std::defaultfloat << convergence_order << ")";
        }
        std::cout << std::endl;

        auto error_field = samurai::make_field<double, 1>("error", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   double x = cell.center(0);
                                   double y = cell.center(1);
                                   // double sol = sin(4 * M_PI * x)*sin(4 * M_PI *
                                   // y); double sol = x * (1 - x) * y*(1 - y);
                                   double sol        = exp(x * y * y);
                                   error_field[cell] = abs(u[cell] - sol);
                               });

        samurai::save(fmt::format("error_ref_{}", ite), mesh, error_field);
        samurai::save(fmt::format("solution_{}_{}_ref_{}", min_level, max_level, ite), mesh, u);
        samurai::save(fmt::format("solution_recons_{}_{}_ref_{}", min_level, max_level, ite), u_recons.mesh(), u_recons);
        h_coarse            = h;
        error_coarse        = error;
        error_recons_coarse = error_recons;
    }
    PetscFinalize();

    return 0;
}
