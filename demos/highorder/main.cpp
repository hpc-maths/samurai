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
    using field_t = Field;
    using Mesh = typename Field::mesh_t;
    static constexpr std::size_t ghost_width = Mesh::config::ghost_width;
    using boundary_condition_t = typename Field::boundary_condition_t;

    HighOrderDiffusion(Field& unknown) : 
        samurai::petsc::CellBasedScheme<cfg, Field>(unknown, stencil(), coefficients)
    {}  

    static constexpr auto stencil()
    {
        return samurai::Stencil<2, 9> {{-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {0, -2}, {0, -1}, {0, 1}, {0, 2}};
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

    static std::array<double, 5> reduced_stencil_coefficients(double h)
    {
        std::array<double, 5> coeffs;
        double one_over_h2 = 1/(h*h);
        coeffs[0] = -4./3  * one_over_h2;
        coeffs[1] =  5.    * one_over_h2;
        coeffs[2] = -4./3  * one_over_h2;
        coeffs[3] = -4./3  * one_over_h2;
        coeffs[4] = -4./3  * one_over_h2;
        return coeffs;
    }

    void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
    {
        auto reduced_stencil = samurai::star_stencil<2>();
        samurai::for_each_stencil_center_and_outside_ghost(this->m_mesh, reduced_stencil, [&](const auto& cells, const auto& towards_ghost)
        {
            auto& cell  = cells[0];
            auto& ghost = cells[1];
            auto boundary_point = cell.face_center(towards_ghost);
            auto bc = find(this->m_boundary_conditions, boundary_point);
            if (bc.is_dirichlet())
            {
                nnz[ghost.index] = 2;
            }
        });
    }

    void assemble_boundary_conditions(Mat& A) override
    {
        auto reduced_stencil = samurai::star_stencil<2>();
        samurai::for_each_stencil_center_and_outside_ghost(this->m_mesh, reduced_stencil, this->reduced_stencil_coefficients, 
        [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
        {
            const auto& cell_init  = cells[0];
            const auto& ghost_init = cells[1];
            auto boundary_point = cell_init.face_center(towards_ghost);
            auto bc = find(this->m_boundary_conditions, boundary_point);

            double coeff = ghost_coeff;
            for (std::size_t ig = 0; ig < ghost_width; ++ig)
            {
                auto cell = cell_init;
                cell.indices += ig*towards_ghost;

                auto ghost = ghost_init;
                ghost.indices -= ig*towards_ghost;

                PetscInt cell_index = static_cast<PetscInt>(this->m_mesh.get_index(cell.level, cell.indices));
                PetscInt ghost_index = static_cast<PetscInt>(this->m_mesh.get_index(ghost.level, ghost.indices));

                if (bc.is_dirichlet())
                {
                    coeff = coeff == 0 ? 1 : coeff;
                    // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [  1/2    1/2 ] = dirichlet_value
                    // which is equivalent to                                                         [-coeff -coeff] = -2 * coeff * dirichlet_value
                    MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                    MatSetValue(A, ghost_index, cell_index , -coeff, INSERT_VALUES);
                }

                this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
            }
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
        auto reduced_stencil = samurai::star_stencil<2>();
        samurai::for_each_stencil_center_and_outside_ghost(this->m_mesh, reduced_stencil, this->reduced_stencil_coefficients,
        [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
        {
            auto& cell  = cells[0];
            auto& ghost = cells[1];
            auto boundary_point = cell.face_center(towards_ghost);
            auto bc = find(this->m_boundary_conditions, boundary_point);

            //PetscInt cell_index = static_cast<PetscInt>(cell.index);
            PetscInt ghost_index = static_cast<PetscInt>(ghost.index);
            double coeff = ghost_coeff;
            if (bc.is_dirichlet())
            {
                double dirichlet_value = bc.get_value(boundary_point);
                coeff = coeff == 0 ? 1 : coeff;
                VecSetValue(b, ghost_index, - 2 * coeff * dirichlet_value, ADD_VALUES);
            }
        });

        // Projection
        samurai::for_each_projection_ghost(this->m_mesh, [&](auto& ghost)
        {
            VecSetValue(b, static_cast<PetscInt>(ghost.index), 0, INSERT_VALUES);
        });

        // Prediction
        for_each_prediction_ghost(this->m_mesh, [&](auto& ghost)
        {
            VecSetValue(b, static_cast<PetscInt>(ghost.index), 0, INSERT_VALUES);
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
