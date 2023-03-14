// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <samurai/hdf5.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/petsc.hpp>


#include <filesystem>
namespace fs = std::filesystem;

// coefficients: https://en.wikipedia.org/wiki/Finite_difference_coefficient


using highOrderStencilFV = samurai::petsc::PetscAssemblyConfig
        <
            1,   // Output field size
            9,   // Stencil size
            2,   // Index of the stencil center
            0, 5 // Start index and size of contiguous cell indices
        >;

template<class Field, class cfg=highOrderStencilFV>
class HighOrderDiffusion : public samurai::petsc::CellBasedScheme<cfg, Field>
{
public:
    static constexpr std::size_t dim = Field::dim;
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    static constexpr std::size_t ghost_width = Mesh::config::ghost_width;
    using boundary_condition_t = typename Field::boundary_condition_t;
    static constexpr std::size_t prediction_order = samurai::petsc::CellBasedScheme<cfg, Field>::prediction_order;

    HighOrderDiffusion(Field& unknown) : 
        samurai::petsc::CellBasedScheme<cfg, Field>(unknown, stencil(), coefficients)
    {}  

    static constexpr auto stencil()
    {
        return samurai::star_stencil<dim, 2>();
    }

    static std::array<double, 9> coefficients(double h)
    {
        std::array<double, 9> coeffs = { 1./12, -4./3, 5., -4./3, 1./12, 1./12, -4./3, -4./3, 1./12 };
        double one_over_h2 = 1/(h*h);
        for (double& coeff : coeffs)
        {
            coeff *= one_over_h2;
        }
        return coeffs;
    }

    void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
    {
        std::array<samurai::StencilVector<dim>, 4> bdry_directions;
        std::array<samurai::Stencil<3, dim>   , 4> bdry_stencils;

        // Left boundary
        bdry_directions[0] = {-1, 0};
        bdry_stencils[0] = {{0, 0}, {-1, 0}, {-2, 0}};
        // Top boundary
        bdry_directions[1] = {0, 1};
        bdry_stencils[1] = {{0, 0}, {0, 1}, {0, 2}};
        // Right boundary
        bdry_directions[2] = {1, 0};
        bdry_stencils[2] = {{0, 0}, {1, 0}, {2, 0}};
        // Bottom boundary
        bdry_directions[3] = {0, -1};
        bdry_stencils[3] = {{0, 0}, {0, -1}, {0, -2}};

        samurai::for_each_stencil_on_boundary(this->m_mesh, bdry_directions, bdry_stencils, [&](const auto& cells, const auto&)
        {
            auto& ghost1 = cells[1];
            auto& ghost2 = cells[2];
            nnz[ghost1.index] = 4;
            nnz[ghost2.index] = 4;
        });
    }

    void assemble_boundary_conditions(Mat& A) override
    {
        std::array<samurai::StencilVector<dim>, 4> bdry_directions;
        std::array<samurai::Stencil<5, dim>   , 4> bdry_stencils;

        // Left boundary
        bdry_directions[0] = {-1, 0};
        bdry_stencils[0] = {{0, 0}, {1, 0}, {2, 0}, {-1, 0}, {-2, 0}};
        // Top boundary
        bdry_directions[1] = {0, 1};
        bdry_stencils[1] = {{0, 0}, {0, -1}, {0, -2}, {0, 1}, {0, 2}};
        // Right boundary
        bdry_directions[2] = {1, 0};
        bdry_stencils[2] = {{0, 0}, {-1, 0}, {-2, 0}, {1, 0}, {2, 0}};
        // Bottom boundary
        bdry_directions[3] = {0, -1};
        bdry_stencils[3] = {{0, 0}, {0, 1}, {0, 2}, {0, -1}, {0, -2}};

        samurai::for_each_stencil_on_boundary(this->m_mesh, bdry_directions, bdry_stencils,
        [&](const auto& cells, const auto&)
        {
            auto& cell1  = cells[0];
            auto& cell2  = cells[1];
            auto& cell3  = cells[2];
            auto& ghost1 = cells[3];
            auto& ghost2 = cells[4];

            PetscInt cell1_index = static_cast<PetscInt>(cell1.index);
            PetscInt cell2_index = static_cast<PetscInt>(cell2.index);
            PetscInt cell3_index = static_cast<PetscInt>(cell3.index);
            PetscInt ghost1_index = static_cast<PetscInt>(ghost1.index);
            PetscInt ghost2_index = static_cast<PetscInt>(ghost2.index);

            // We need to define a polynomial of degree 3 that passes by the 4 points c3, c2, c1 and the boundary point.
            // This polynomial writes 
            //                       p(x) = a*x^2 + b*x^2 + c*x + d.
            // The coefficients a, b, c, d are found by inverting the Vandermonde matrix obtained by inserting the 4 point into the polynomial.
            // If we set the abscissa 0 at the center of c1, this system reads
            //                       p(  0) = u1
            //                       p( -h) = u2
            //                       p(-2h) = u3
            //                       p(h/2) = dirichlet_value.
            // Then, we want that the ghost values be also located on this polynomial, i.e.
            //                       u_g1 = p( -h)
            //                       u_g2 = p(-2h).
            // This gives
            //             5/16 * u_g1  +  15/16 * u1  -5/16 * u2  +  1/16 * u3 = dirichlet_value
            //             5/64 * u_g2  +  45/32 * u1  -5/8  * u2  +  9/64 * u3 = dirichlet_value 
            
            MatSetValue(A, ghost1_index, ghost1_index,  5./16, INSERT_VALUES);
            MatSetValue(A, ghost1_index, cell1_index,  15./16, INSERT_VALUES);
            MatSetValue(A, ghost1_index, cell2_index , -5./16, INSERT_VALUES);
            MatSetValue(A, ghost1_index, cell3_index ,  1./16, INSERT_VALUES);

            MatSetValue(A, ghost2_index, ghost2_index,  5./64, INSERT_VALUES);
            MatSetValue(A, ghost2_index, cell1_index,  45./32, INSERT_VALUES);
            MatSetValue(A, ghost2_index, cell2_index , -5./8 , INSERT_VALUES);
            MatSetValue(A, ghost2_index, cell3_index ,  9./64, INSERT_VALUES);

            this->m_is_row_empty[ghost1.index] = false;
            this->m_is_row_empty[ghost2.index] = false;

        });
    }

    void enforce_bc(Vec& b) const override
    {
        std::array<samurai::StencilVector<dim>, 4> bdry_directions;
        std::array<samurai::Stencil<3, dim>   , 4> bdry_stencils;

        // Left boundary
        bdry_directions[0] = {-1, 0};
        bdry_stencils[0] = {{0, 0}, {-1, 0}, {-2, 0}};
        // Top boundary
        bdry_directions[1] = {0, 1};
        bdry_stencils[1] = {{0, 0}, {0, 1}, {0, 2}};
        // Right boundary
        bdry_directions[2] = {1, 0};
        bdry_stencils[2] = {{0, 0}, {1, 0}, {2, 0}};
        // Bottom boundary
        bdry_directions[3] = {0, -1};
        bdry_stencils[3] = {{0, 0}, {0, -1}, {0, -2}};

        samurai::for_each_stencil_on_boundary(this->m_mesh, bdry_directions, bdry_stencils,
        [&](const auto& cells, const auto& towards_ghost)
        {
            auto& cell1  = cells[0];
            auto& ghost1 = cells[1];
            auto& ghost2 = cells[2];

            PetscInt ghost1_index = static_cast<PetscInt>(ghost1.index);
            PetscInt ghost2_index = static_cast<PetscInt>(ghost2.index);

            auto boundary_point = cell1.face_center(towards_ghost);
            auto bc = samurai::find(this->m_boundary_conditions, boundary_point);
            double dirichlet_value = bc.get_value(boundary_point);

            VecSetValue(b, ghost1_index, dirichlet_value, ADD_VALUES);
            VecSetValue(b, ghost2_index, dirichlet_value, ADD_VALUES);
        });
    }

    void assemble_prediction(Mat& A) override
    {
        using index_t = int;
        constexpr std::size_t field_size = 1;

        samurai::for_each_prediction_ghost(this->m_mesh, [&](auto& ghost)
        {
            for (unsigned int field_i = 0; field_i < field_size; ++field_i)
            {
                PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                auto ii = ghost.indices(0);
                auto ig = ii>>1;
                auto  j = ghost.indices(1);
                auto jg = j>>1;
                double isign = (ii & 1)? -1.: 1.;
                double jsign = (j & 1)? -1.: 1.;

                auto interpx = samurai::interp_coeffs<2*prediction_order+1>(isign);
                auto interpy = samurai::interp_coeffs<2*prediction_order+1>(jsign);

                auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig, jg)), field_i);
                MatSetValue(A, ghost_index, parent_index, -1, INSERT_VALUES);

                // std::cout << fmt::format("level: {}, i: {}, j: {} pred_cell: ", ghost.level, ii, j);
                for(std::size_t ci = 0; ci < interpx.size(); ++ci)
                {
                    for(std::size_t cj = 0; cj < interpy.size(); ++cj)
                    {
                        if (ci != prediction_order || cj != prediction_order)
                        {
                            double value = -interpx[ci]*interpy[cj];
                            // std::cout << fmt::format("({}, {}, {}) ", ghost.level-1, ig + static_cast<index_t>(ci - order), jg + static_cast<index_t>(cj - order));
                            parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig + static_cast<index_t>(ci - prediction_order), jg + static_cast<index_t>(cj - prediction_order))), field_i);
                            MatSetValue(A, ghost_index, parent_index, value, INSERT_VALUES);
                        }
                    }
                }
                // std::cout << std::endl;
                this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
            }
            // std::cout << std::endl;
        });
    }
};


template<class Field>
auto make_high_order_diffusion(Field& f)
{
    return HighOrderDiffusion<Field>(f);
}

int main(int argc, char *argv[])
{
    constexpr std::size_t dim = 2;
    constexpr std::size_t stencil_width = 2;
    constexpr std::size_t graduation_width = 4;
    constexpr std::size_t max_refinement_level = 20;
    constexpr std::size_t prediction_order = 3;
    using Config = samurai::MRConfig<dim,
                                     stencil_width,
                                     graduation_width,
                                     max_refinement_level,
                                     prediction_order
    >;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.}, max_corner = {1., 1.};

    // Multiresolution parameters
    std::size_t min_level = 4, max_level = 4;
    std::size_t refinement = 0;
    double mr_epsilon = 2.e-4; // Threshold used by multiresolution
    double mr_regularity = 1.; // Regularity guess for multiresolution
    bool correction = false;

    // Output parameters
    fs::path path = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles = 1;

    CLI::App app{"Finite volume example for the advection equation in 2d using multiresolution"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", min_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--refinement", refinement, "Number of refinement")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles,  "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);
    
    samurai::Box<double, dim> box(min_corner, max_corner);
    using mesh_t = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type = typename mesh_t::cl_type;
    mesh_t mesh{box, min_level, max_level};

    PetscInitialize(&argc, &argv, 0, nullptr);
    PetscOptionsSetValue(NULL, "-options_left", "off");

    auto adapt_field = samurai::make_field<double, 1>("adapt_field", mesh, [](const auto& coord) 
            { 
                const auto& x = coord[0];
                const auto& y = coord[1];
                double radius = 0.1;
                if ((x -0.5)*(x-0.5) + (y -0.5)*(y-0.5) < radius*radius)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }, 0);

    
    // std::array<samurai::StencilVector<dim>, 4> bdry_directions;
    // std::array<samurai::Stencil<5, dim>   , 4> bdry_stencils;

    // // Left boundary
    // bdry_directions[0] = {-1, 0};
    // bdry_stencils[0] = {{0, 0}, {1, 0}, {2, 0}, {-1, 0}, {-2, 0}};
    // // Top boundary
    // bdry_directions[1] = {0, 1};
    // bdry_stencils[1] = {{0, 0}, {0, -1}, {0, -2}, {0, 1}, {0, 2}};
    // // Right boundary
    // bdry_directions[2] = {1, 0};
    // bdry_stencils[2] = {{0, 0}, {-1, 0}, {-2, 0}, {1, 0}, {2, 0}};
    // // Bottom boundary
    // bdry_directions[3] = {0, -1};
    // bdry_stencils[3] = {{0, 0}, {0, 1}, {0, 2}, {0, -1}, {0, -2}};

    auto update_bc = [&](auto& , std::size_t)
    {
        // samurai::for_each_stencil_on_boundary(field.mesh(), bdry_directions, bdry_stencils,
        // [&](const auto& cells, const auto& towards_ghost)
        // {
        //     auto& cell1  = cells[0];
        //     auto& cell2  = cells[1];
        //     auto& cell3  = cells[2];
        //     auto& ghost1 = cells[3];
        //     auto& ghost2 = cells[4];
        //     const double& h = cell1.length;
        //     auto boundary_point = cell1.face_center(towards_ghost);
        //     auto bc = find(field.boundary_conditions(), boundary_point);
        //     auto dirichlet_value = bc.get_value(boundary_point);

        //     field[ghost1] =  -3 * field[cell1] +      field[cell2] - 1./5 * field[cell3] + 16./5 * dirichlet_value;
        //     field[ghost2] = -18 * field[cell1] + 8  * field[cell2] - 9./5 * field[cell3] + 64./5 * dirichlet_value;
        // });
    };
    
    auto MRadaptation = samurai::make_MRAdapt(adapt_field, update_bc);
    MRadaptation(mr_epsilon, mr_regularity);

    // samurai::save("initial_mesh", mesh);

    // for(std::size_t ite = 0; ite < refinement; ++ite)
    // {
    //     cl_type cl;
    //     samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
    //     {
    //         samurai::static_nested_loop<dim-1, 0, 2>([&](auto& stencil)
    //         {
    //             auto new_index = 2*index + stencil;
    //             cl[level+1][new_index].add_interval(i<<1);
    //         });
    //     });
    //     mesh = {cl, mesh.min_level(), mesh.max_level()+1};
    // }
    // samurai::save("refine_mesh", mesh);
    
    // Equation: -Lap u = f   in [0, 1]^2
    //            f(x,y) = 2(y(1-y) + x(1-x))
    auto f = samurai::make_field<double, 1>("f", mesh, [](const auto& coord) 
            { 
                const auto& x = coord[0];
                const auto& y = coord[1];
                //return 2 * (y*(1 - y) + x * (1 - x));
                //return 2 * pow(4 * M_PI, 2) * sin(4 * M_PI * x)*sin(4 * M_PI * y);
                return (-pow(y, 4) - 2 * x*(1 + 2 * x*y*y))*exp(x*y*y);
            }, 0);

    // samurai::for_each_cell(mesh[mesh_id_t::reference], [&](auto& cell)
    // {
    //     double x = cell.center(0);
    //     double y = cell.center(1);
    //     f[cell] =  2 * (y*(1 - y) + x * (1 - x));
    // });

    // std::size_t level = mesh.max_level();
    // auto set = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level-1]).on(level-1);
    // set.apply_op(samurai::projection(f));
    // auto f_recons = samurai::make_field<double, 1>("f_recons", mesh);
    // auto error_f = samurai::make_field<double, 1>("error", mesh);
    // set.apply_op(samurai::prediction<prediction_order, true>(f_recons, f));
    // samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
    // {
    //     auto j = index[0];
    //     error_f(level, i, j) = xt::abs(f(level, i, j) - f_recons(level, i, j));
    // });
    // samurai::save("test_pred", mesh, f, f_recons, error_f);
    // return 0;

    auto u = samurai::make_field<double, 1>("u", mesh);
    u.set_dirichlet([](const auto& coord) 
        { 
            const auto& x = coord[0];
            const auto& y = coord[1];
            //return 0.;
            return exp(x*y*y);
        }).everywhere();
    u.fill(0);

    auto diff = make_high_order_diffusion(u);

    auto solver = samurai::petsc::make_solver(diff);
    solver.solve(f);

    double error = diff.L2Error(u, [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                //return x * (1 - x) * y*(1 - y);
                //return sin(4 * M_PI * x)*sin(4 * M_PI * y);
                return exp(x*y*y);
            });
    std::cout.precision(2);
    std::cout << "L2-error: " << std::scientific << error << std::endl;

    auto error_field = samurai::make_field<double, 1>("error", mesh);
    samurai::for_each_cell(mesh, [&](const auto& cell) 
    { 
        double x = cell.center(0);
        double y = cell.center(1);
        //double sol = sin(4 * M_PI * x)*sin(4 * M_PI * y);
        //double sol = x * (1 - x) * y*(1 - y);
        double sol = exp(x*y*y);
        error_field[cell] = abs(u[cell]-sol); 
    });

    samurai::save("error", mesh, error_field);

    samurai::save("solution", mesh, u);


    // Destroy Petsc objects
    solver.destroy_petsc_objects();
    PetscFinalize();

    return 0; 
}
