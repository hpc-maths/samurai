// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field,
          // scheme config
          std::size_t neighbourhood_width = 2,
          class cfg      = samurai::StarStencilSchemeConfig<samurai::SchemeType::LinearHomogeneous, neighbourhood_width, Field, Field>,
          class bdry_cfg = samurai::BoundaryConfigFV<neighbourhood_width>>
class HighOrderDiffusion : public samurai::CellBasedScheme<cfg, bdry_cfg>
{
  public:

    static constexpr std::size_t dim = Field::dim;

    HighOrderDiffusion()
    {
        this->stencil()           = samurai::star_stencil<dim, neighbourhood_width>();
        this->coefficients_func() = [](samurai::StencilCoeffs<cfg>& coeffs, double h)
        {
            //        left2,    left, center, right, right2   bottom2,  bottom,   top,    top2
            coeffs = {1. / 12, -4. / 3, 5., -4. / 3, 1. / 12, 1. / 12, -4. / 3, -4. / 3, 1. / 12};
            coeffs /= (h * h);
        };
        set_dirichlet_config();
    }

    void set_dirichlet_config()
    {
        for (std::size_t d = 0; d < 2 * dim; ++d)
        {
            auto& config = this->dirichlet_config()[d];

            // Example of directional stencil:
            //
            // direction:    {1,0} (--> right)
            // stencil  :      [0]     [1]     [2]    [3]    [4]
            //               {{0,0}, {-1,0}, {-2,0}, {1,0}, {2,0}};
            //
            //                 [2]     [1]     [0]     [3]     [4]
            //              |_______|_______|_______|.......|.......|
            //                cell3   cell2   cell1  ghost1  ghost2

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
        }
    }
};

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 2d using multiresolution", argc, argv);

    constexpr std::size_t dim              = 2;
    constexpr std::size_t stencil_width    = 2;
    constexpr std::size_t graduation_width = 4;
    constexpr std::size_t prediction_order = 1;
    using Config                           = samurai::MRConfig<dim, stencil_width, graduation_width, prediction_order>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

    // Multiresolution parameters
    std::size_t min_level  = 4;
    std::size_t max_level  = 4;
    std::size_t refinement = 5;
    bool correction        = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "poisson_highorder_2d";
    std::size_t nfiles   = 1;

    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", min_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--refinement", refinement, "Number of refinement")->capture_default_str()->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    samurai::Box<double, dim> box(min_corner, max_corner);
    using mesh_t    = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type   = typename mesh_t::cl_type;

    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    mesh_t init_mesh{config, box};

    auto adapt_field = samurai::make_scalar_field<double>("adapt_field",
                                                          init_mesh,
                                                          [](const auto& coord)
                                                          {
                                                              const auto& x = coord[0];
                                                              const auto& y = coord[1];
                                                              double radius = 0.4;
                                                              if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < radius * radius)
                                                              {
                                                                  samurai::finalize();
                                                                  return 0;
                                                              }
                                                              else
                                                              {
                                                                  return 1;
                                                              }
                                                          });

    auto MRadaptation = samurai::make_MRAdapt(adapt_field);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    MRadaptation(mra_config);

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
                                               [&](const auto& stencil)
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
        auto f = samurai::make_scalar_field<double>("f",
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
        // auto f_recons = samurai::make_scalar_field<double>("f_recons", mesh);
        // auto error_f = samurai::make_scalar_field<double>("error", mesh);
        // set.apply_op(samurai::prediction<prediction_order, true>(f_recons, f));
        // samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level,
        // const auto& i, const auto& index)
        // {
        //     auto j = index[0];
        //     error_f(level, i, j) = xt::abs(f(level, i, j) - f_recons(level, i,
        //     j));
        // });
        // samurai::save("test_pred", mesh, f, f_recons, error_f);
        // samurai::finalize();

        auto u = samurai::make_scalar_field<double>("u", mesh);
        samurai::make_bc<samurai::Dirichlet<2>>(u,
                                                [](const auto&, const auto&, const auto& coord)
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

        double h = mesh.cell_length(mesh.min_level());

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

        auto error_field = samurai::make_scalar_field<double>("error", mesh);
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

    samurai::finalize();
    return 0;
}
