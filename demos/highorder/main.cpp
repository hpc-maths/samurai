// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <samurai/hdf5.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>

#include <filesystem>
namespace fs = std::filesystem;

// coefficients: https://en.wikipedia.org/wiki/Finite_difference_coefficient

 using highOrderStencilFV = samurai::petsc::cellBasedConfig
        <
            2, //dim
            1, //output_field_size,
            // ----  Stencil size 
            // Only one cell:
            9,
            // ---- Index of the stencil center
            2, 
            // ---- Start index and size of contiguous cell indices
            0, 5
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

    HighOrderDiffusion(Field& unknown) : 
        samurai::petsc::CellBasedScheme<cfg, Field>(unknown, stencil(), coefficients)
    {}  

    static constexpr auto stencil()
    {
        return samurai::Stencil<dim, 9> {{-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {0, -2}, {0, -1}, {0, 1}, {0, 2}};
    }

    static std::array<double, 9> coefficients(double h)
    {
        std::array<double, 9> coeffs;
        double one_over_h2 = 1/(h*h);
        coeffs[0] =  1./12 * one_over_h2;
        coeffs[1] = -4./3  * one_over_h2;
        coeffs[2] =  5.    * one_over_h2;
        coeffs[3] = -4./3  * one_over_h2;
        coeffs[4] =  1./12 * one_over_h2;
        coeffs[5] =  1./12 * one_over_h2;
        coeffs[6] = -4./3  * one_over_h2;
        coeffs[7] = -4./3  * one_over_h2;
        coeffs[8] =  1./12 * one_over_h2;
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
            nnz[ghost1.index] = 2;
            nnz[ghost2.index] = 2;
        });
    }

    void assemble_boundary_conditions(Mat& A) override
    {
        std::array<samurai::StencilVector<dim>, 4> bdry_directions;
        std::array<samurai::Stencil<4, dim>   , 4> bdry_stencils;

        // Left boundary
        bdry_directions[0] = {-1, 0};
        bdry_stencils[0] = {{0, 0}, {1, 0}, {-1, 0}, {-2, 0}};
        // Top boundary
        bdry_directions[1] = {0, 1};
        bdry_stencils[1] = {{0, 0}, {0, -1}, {0, 1}, {0, 2}};
        // Right boundary
        bdry_directions[2] = {1, 0};
        bdry_stencils[2] = {{0, 0}, {-1, 0}, {1, 0}, {2, 0}};
        // Bottom boundary
        bdry_directions[3] = {0, -1};
        bdry_stencils[3] = {{0, 0}, {0, 1}, {0, -1}, {0, -2}};

        samurai::for_each_stencil_on_boundary(this->m_mesh, bdry_directions, bdry_stencils,
        [&](const auto& cells, const auto&)
        {
            auto& cell1  = cells[0];
            auto& cell2  = cells[1];
            auto& ghost1 = cells[2];
            auto& ghost2 = cells[3];

            PetscInt cell1_index = static_cast<PetscInt>(cell1.index);
            PetscInt cell2_index = static_cast<PetscInt>(cell2.index);
            PetscInt ghost1_index = static_cast<PetscInt>(ghost1.index);
            PetscInt ghost2_index = static_cast<PetscInt>(ghost2.index);

            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [1/2 1/2] = dirichlet_value
            MatSetValue(A, ghost1_index, ghost1_index, 0.5, INSERT_VALUES);
            MatSetValue(A, ghost1_index, cell1_index , 0.5, INSERT_VALUES);
            this->m_is_row_empty[ghost1.index] = false;

            MatSetValue(A, ghost2_index, ghost2_index, 0.5, INSERT_VALUES);
            MatSetValue(A, ghost2_index, cell2_index , 0.5, INSERT_VALUES);
            this->m_is_row_empty[ghost2.index] = false;
        });


        // Add 1 on the diagonal for the unused outside ghosts
        if (this->m_add_1_on_diag_for_useless_ghosts)
        {
            samurai::for_each_outside_ghost(this->m_mesh, [&](const auto& ghost)
            {
                auto ghost_row = static_cast<PetscInt>(ghost.index);
                if (this->m_is_row_empty[static_cast<std::size_t>(ghost_row)])
                {
                    MatSetValue(A, ghost_row, ghost_row, 1, INSERT_VALUES);
                    this->m_is_row_empty[static_cast<std::size_t>(ghost_row)] = false;
                }
            });

            // For some reason, there might be unused inside ghosts
            for (std::size_t i = 0; i<this->m_is_row_empty.size(); i++)
            {
                if (this->m_is_row_empty[i])
                {
                    MatSetValue(A, i, i, 1, INSERT_VALUES);
                    this->m_is_row_empty[i] = false;
                }
            }
        }
    }

    void enforce_bc(Vec& b) const override
    {
        std::array<samurai::StencilVector<dim>, 4> bdry_directions;
        std::array<samurai::Stencil<4, dim>   , 4> bdry_stencils;

        // Left boundary
        bdry_directions[0] = {-1, 0};
        bdry_stencils[0] = {{0, 0}, {1, 0}, {-1, 0}, {-2, 0}};
        // Top boundary
        bdry_directions[1] = {0, 1};
        bdry_stencils[1] = {{0, 0}, {0, -1}, {0, 1}, {0, 2}};
        // Right boundary
        bdry_directions[2] = {1, 0};
        bdry_stencils[2] = {{0, 0}, {-1, 0}, {1, 0}, {2, 0}};
        // Bottom boundary
        bdry_directions[3] = {0, -1};
        bdry_stencils[3] = {{0, 0}, {0, 1}, {0, -1}, {0, -2}};

        samurai::for_each_stencil_on_boundary(this->m_mesh, bdry_directions, bdry_stencils,
        [&](const auto& cells, const auto& towards_ghost)
        {
            auto& cell1  = cells[0];
            //auto& cell2  = cells[1];
            auto& ghost1 = cells[2];
            auto& ghost2 = cells[3];

            PetscInt ghost1_index = static_cast<PetscInt>(ghost1.index);
            PetscInt ghost2_index = static_cast<PetscInt>(ghost2.index);

            auto boundary_point = cell1.face_center(towards_ghost);
            auto bc = samurai::find(this->m_boundary_conditions, boundary_point);
            double dirichlet_value = bc.get_value(boundary_point);
            
            VecSetValue(b, ghost1_index, dirichlet_value, ADD_VALUES);
            VecSetValue(b, ghost2_index, dirichlet_value, ADD_VALUES);
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
    using Config = samurai::MRConfig<dim, stencil_width>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.}, max_corner = {1., 1.};

    // Multiresolution parameters
    std::size_t min_level = 4, max_level = 4;
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
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles,  "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);
    
    samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    PetscInitialize(&argc, &argv, 0, nullptr);
    PetscOptionsSetValue(NULL, "-options_left", "off");

    
    // Equation: -Lap u = f   in [0, 1]^2
    //            f(x,y) = 2(y(1-y) + x(1-x))
    auto f = samurai::make_field<double, 1>("f", mesh, [](const auto& coord) 
            { 
                const auto& x = coord[0];
                const auto& y = coord[1];
                return 2 * (y*(1 - y) + x * (1 - x));
            }, 0);
    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0);

    u.set_dirichlet([](const auto&) { return 0.; }).everywhere();

    auto diff = make_high_order_diffusion(u);

    auto solver = samurai::petsc::make_solver(diff);
    solver.solve(f);

    double error = diff.L2Error(u, [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                return x * (1 - x) * y*(1 - y);
            });
    std::cout.precision(2);
    std::cout << "L2-error: " << std::scientific << error << std::endl;


    // Destroy Petsc objects
    solver.destroy_petsc_objects();
    PetscFinalize();

    return 0; 
}
