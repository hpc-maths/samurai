// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

using aux_t = xt::xtensor<double, 2>;

template <class Field_e>
struct Coupling_auxCe_e : public samurai::petsc::ManualAssembly</* output */ Field_e, /* input */ aux_t>
{
    using Mesh_e = typename Field_e::mesh_t;

    const Mesh_e* mesh_e; // use a pointer instead of a reference to avoid issues with the copy constructor

    explicit Coupling_auxCe_e(const Mesh_e& m)
        : mesh_e(&m)
    {
        this->set_name("Coupling_auxCe_e");
    }

    PetscInt local_matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_e->cell_ownership().n_local_cells);
    }

    PetscInt local_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt owned_matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_e->cell_ownership().n_owned_cells);
    }

    PetscInt owned_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& /*o_nnz*/) const override
    {
        samurai::for_each_boundary_interface__direction(*mesh_e,
                                                        {1, 0},
                                                        [&](auto& cell, auto&)
                                                        {
                                                            std::size_t row = static_cast<std::size_t>(this->block_row_shift() + cell.index);
                                                            d_nnz[row] += 2;
                                                        });
    }

    void assemble_scheme(Mat& A) override
    {
        PetscInt i = 0;
        samurai::for_each_boundary_interface__direction(*mesh_e,
                                                        {1, 0},
                                                        [&](auto& cell, auto&)
                                                        {
                                                            PetscInt row = this->block_row_shift() + static_cast<PetscInt>(cell.index);
                                                            PetscInt col = this->block_col_shift() + i;
                                                            double coeff = 123;                                   // random...
                                                            MatSetValueLocal(A, row, col, coeff, ADD_VALUES);     // 1st aux variable
                                                            MatSetValueLocal(A, row, col + 1, coeff, ADD_VALUES); // 2nd aux variable
                                                            i += 2;
                                                        });
    }
};

template <class Field_e>
struct Coupling_e_auxCe : public samurai::petsc::ManualAssembly</* output */ aux_t, /* input */ Field_e>
{
    const aux_t* aux_Ce;

    explicit Coupling_e_auxCe(const aux_t& t)
        : aux_Ce(&t)
    {
        this->set_name("Coupling_e_auxCe");
    }

    PetscInt local_matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce->size());
    }

    PetscInt local_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().cell_ownership().n_local_cells);
    }

    PetscInt owned_matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce->size());
    }

    PetscInt owned_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().cell_ownership().n_owned_cells);
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

struct Coupling_auxCe_auxCe : public samurai::petsc::ManualAssembly<aux_t, aux_t>
{
    Coupling_auxCe_auxCe()
    {
        this->set_name("Coupling_auxCe_auxCe");
    }

    PetscInt local_matrix_rows() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt local_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt owned_matrix_rows() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt owned_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

template <class Field_s>
struct Coupling_s_auxCe : public samurai::petsc::ManualAssembly</* output */ aux_t, /* input */ Field_s>
{
    const aux_t* aux_Ce;

    explicit Coupling_s_auxCe(const aux_t& t)
        : aux_Ce(&t)
    {
        this->set_name("Coupling_s_auxCe");
    }

    PetscInt local_matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce->size());
    }

    PetscInt local_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().cell_ownership().n_local_cells);
    }

    PetscInt owned_matrix_rows() const override
    {
        return static_cast<PetscInt>(aux_Ce->size());
    }

    PetscInt owned_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().mesh().cell_ownership().n_owned_cells);
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
    {
        // TODO
    }

    void assemble_scheme(Mat&) override
    {
        // TODO
    }
};

template <class Field_s>
struct Coupling_auxCe_s : public samurai::petsc::ManualAssembly</* output */ Field_s, /* input */ aux_t>
{
    using Mesh_s = typename Field_s::mesh_t;

    const Mesh_s* mesh_s;

    explicit Coupling_auxCe_s(const Mesh_s& m)
        : mesh_s(&m)
    {
        this->set_name("Coupling_auxCe_s");
    }

    PetscInt local_matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_s->cell_ownership().n_local_cells);
    }

    PetscInt local_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    PetscInt owned_matrix_rows() const override
    {
        return static_cast<PetscInt>(mesh_s->cell_ownership().n_owned_cells);
    }

    PetscInt owned_matrix_cols() const override
    {
        return static_cast<PetscInt>(this->unknown().size());
    }

    void sparsity_pattern_scheme(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
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
    using Box                        = samurai::Box<double, dim>;

    std::cout << "------------------------- Begin -------------------------" << std::endl;

    Box box({0, 0}, {1, 1});
    auto mesh_cfg = samurai::mesh_config<dim>().min_level(3).max_level(3);
    auto mesh_e   = samurai::mra::make_mesh(box, mesh_cfg);
    auto mesh_s   = samurai::mra::make_mesh(box, mesh_cfg);

    //-------------------------------//
    // Fields and auxiliary unknowns //
    //-------------------------------//

    auto u_e = samurai::make_scalar_field<double>("u_e", mesh_e, 0);
    auto u_s = samurai::make_scalar_field<double>("u_s", mesh_s, 1);

    using Field_e = decltype(u_e);
    using Field_s = decltype(u_s);

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
    Coupling_auxCe_e<Field_e> auxCe_e(mesh_e);
    Coupling_e_auxCe<Field_e> e_auxCe(aux_Ce);
    Coupling_auxCe_auxCe auxCe_auxCe;
    Coupling_s_auxCe<Field_s> s_auxCe(aux_Ce);
    Coupling_auxCe_s<Field_s> auxCe_s(mesh_s);

    // Define the block operator

    // clang-format off
    auto block_op = samurai::make_block_operator<3, 3>(id + diff_e,  auxCe_e,        0,
                                                           e_auxCe, auxCe_auxCe, s_auxCe,
                                                               0,    auxCe_s,     diff_s);
    // clang-format on

    //-----------------//
    // Matrix assembly //
    //-----------------//

    // Create an assembly object in order to assemble the matrix associated to the block operator
    auto assembly = samurai::petsc::make_assembly<samurai::petsc::BlockAssemblyType::Monolithic>(block_op);
    // Disable the assembly of the BC for the diffusion operators
    assembly.template get<0, 0>().include_bc(false);
    assembly.template get<2, 2>().include_bc(false);
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

    Vec v = assembly.create_vector(u_e, aux_Ce, u_s);
    VecView(v, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
    std::cout << std::endl;

    // Just to check that it compiles
    auto solver = samurai::petsc::make_solver(block_op);
    solver.set_unknowns(u_e, aux_Ce, u_s);
    solver.set_block_operator(block_op);

    samurai::finalize();
    return 0;
}
