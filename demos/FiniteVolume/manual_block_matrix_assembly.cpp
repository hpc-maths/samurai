// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using aux_t = xt::xtensor<double, 2>;

template <class Mesh_e>
struct Coupling_auxCe_e : public samurai::petsc::ManualAssembly<aux_t> // <...>: type of field the block applies to
                                                                       // (= unknown field type if the block must be inversed)
{
    const Mesh_e& mesh_e;

    explicit Coupling_auxCe_e(const Mesh_e& m)
        : mesh_e(m)
    {
        this->set_name("Coupling_auxCe_e");
    }

    PetscInt matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_e.nb_cells());
    }

    PetscInt matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
    {
        samurai::for_each_boundary_interface__direction(mesh_e,
                                                        {1, 0},
                                                        [&](auto& cell, auto&)
                                                        {
                                                            std::size_t row = static_cast<std::size_t>(this->row_shift() + cell.index);
                                                            nnz[row] += 2;
                                                        });
    }

    void assemble_scheme(Mat& A) override
    {
        PetscInt i = 0;
        samurai::for_each_boundary_interface__direction(mesh_e,
                                                        {1, 0},
                                                        [&](auto& cell, auto&)
                                                        {
                                                            PetscInt row = this->row_shift() + static_cast<PetscInt>(cell.index);
                                                            PetscInt col = this->col_shift() + i;
                                                            double coeff = 123;                              // random...
                                                            MatSetValue(A, row, col, coeff, ADD_VALUES);     // 1st aux variable
                                                            MatSetValue(A, row, col + 1, coeff, ADD_VALUES); // 2nd aux variable
                                                            i += 2;
                                                        });
    }
};

template <class field_t>
struct Coupling_e_auxCe : public samurai::petsc::ManualAssembly<field_t>
{
    const aux_t& aux_Ce;

    explicit Coupling_e_auxCe(const aux_t& t)
        : aux_Ce(t)
    {
        this->set_name("Coupling_e_auxCe");
    }

    PetscInt matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce.size());
    }

    PetscInt matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().nb_cells());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

struct Coupling_auxCe_auxCe : public samurai::petsc::ManualAssembly<aux_t>
{
    Coupling_auxCe_auxCe()
    {
        this->set_name("Coupling_auxCe_auxCe");
    }

    PetscInt matrix_rows() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

template <class field_t>
struct Coupling_s_auxCe : public samurai::petsc::ManualAssembly<field_t>
{
    const aux_t* aux_Ce;

    explicit Coupling_s_auxCe(const aux_t& t)
        : aux_Ce(&t)
    {
        this->set_name("Coupling_s_auxCe");
    }

    PetscInt matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce->size());
    }

    PetscInt matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().nb_cells());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

template <class Mesh_s>
struct Coupling_auxCe_s : public samurai::petsc::ManualAssembly<aux_t>
{
    const Mesh_s& mesh_s;

    explicit Coupling_auxCe_s(const Mesh_s& m)
        : mesh_s(m)
    {
        this->set_name("Coupling_auxCe_s");
    }

    PetscInt matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_s.nb_cells());
    }

    PetscInt matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;

    std::cout << "------------------------- Begin -------------------------" << std::endl;

    std::size_t min_level = 3;
    std::size_t max_level = 3;

    Box box({0, 0}, {1, 1});
    auto mesh_cfg = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    samurai::MRMesh<Config> mesh_e{mesh_cfg, box};
    samurai::MRMesh<Config> mesh_s{mesh_cfg, box};

    //-------------------------------//
    // Fields and auxiliary unknowns //
    //-------------------------------//

    auto u_e = samurai::make_scalar_field<double>("u_e", mesh_e, 0);
    auto u_s = samurai::make_scalar_field<double>("u_s", mesh_s, 1);

    // Count the number of cells at the interface between mesh_e and mesh_s
    std::size_t n_interface_cells = 0;
    samurai::for_each_boundary_interface__direction(mesh_e,
                                                    {1, 0}, // right boundary of mesh_e
                                                    [&](auto&, auto&)
                                                    {
                                                        n_interface_cells++;
                                                    });

    // Auxiliary values
    aux_t::shape_type shape = {n_interface_cells, 2};
    aux_t aux_Ce            = xt::zeros<double>(shape);

    xt::view(aux_Ce, xt::all(), 0) = 1.;
    xt::view(aux_Ce, xt::all(), 1) = 0.;

    // samurai::make_bc<samurai::Neumann<1>>(u_s, 0.);
    // samurai::make_bc<samurai::Neumann<1>>(u_e, 0.);

    //---------------------------------------------------------//
    // Diffusion operators and definition of the matrix blocks //
    //---------------------------------------------------------//

    // Diffusion operators for the electrolyte and the solid
    double D_e  = 1; // diffusion coefficient
    auto diff_e = samurai::make_diffusion_order2<decltype(u_e)>(D_e);
    diff_e.include_boundary_fluxes(false);
    double D_s  = 2;
    auto diff_s = samurai::make_diffusion_order2<decltype(u_s)>(D_s);
    auto id     = samurai::make_identity<decltype(u_e)>();

    // Definition of the matrix blocks for the couplings to the auxiliary values
    Coupling_auxCe_e auxCe_e(mesh_e);
    Coupling_e_auxCe<decltype(u_e)> e_auxCe(aux_Ce);
    Coupling_auxCe_auxCe auxCe_auxCe;
    Coupling_s_auxCe<decltype(u_s)> s_auxCe(aux_Ce);
    Coupling_auxCe_s auxCe_s(mesh_s);

    // Define the block operator

    // clang-format off
    auto block_op = samurai::make_block_operator<3, 3>(id + diff_e,  auxCe_e,        0,     // simply put 0 for zero-blocks
                                                           e_auxCe, auxCe_auxCe, s_auxCe,
                                                               0,    auxCe_s,     diff_s);
    // clang-format on

    //-----------------//
    // Matrix assembly //
    //-----------------//

    // Create an assembly object in order to assemble the matrix associated to the block operator
    auto assembly = samurai::petsc::make_assembly<true>(block_op); // <true>: monolithic, <false>: nested
    // Disable the assembly of the BC for the diffusion operators
    assembly.get<0, 0>().include_bc(false);
    assembly.get<2, 2>().include_bc(false);
    assembly.set_diag_value_for_useless_ghosts(9);

    // Set the unknowns of the system (even if you don't want the solve it).
    // They are used to determine the size of each block, and to perform some compatibility checks.
    assembly.set_unknowns(u_e, aux_Ce, u_s);

    // Declare the Jacobian matrix
    Mat J;
    // Allocate the matrix (number of non-zero coefficients per row)
    assembly.create_matrix(J);
    // Insert the coefficients into the matrix
    assembly.assemble_matrix(J);

    std::cout << "Useless ghost rows: ";
    // assembly.get<0, 0>().for_each_useless_ghost_row(
    assembly.for_each_useless_ghost_row(
        [](auto row)
        {
            std::cout << row << " ";
        });
    std::cout << std::endl;

    Vec v = assembly.create_vector(u_e, aux_Ce, u_s);
    VecView(v, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
    std::cout << std::endl;

    samurai::finalize();
    return 0;
}
