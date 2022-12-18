#pragma once
#include "petsc_cell_based_scheme_assembly.hpp"

namespace samurai { namespace petsc
{
    /**
     * @class PetscBackwardEuler
    */
    template<class Operator>
    class PetscBackwardEuler : public PetscAssembly
    {
    public:
        using field_t = typename Operator::field_t;
        using Mesh = typename field_t::mesh_t;
    private:
        const Operator& _operator;
        double _dt;

    public:
        PetscBackwardEuler(Operator& op, double dt) : 
            _operator(op),
            _dt(dt)
        {}

        auto& mesh() const
        {
            return _operator.mesh();
        }

        PetscInt matrix_size() const override
        {
            return _operator.matrix_size();
        }

        std::vector<PetscInt> sparsity_pattern() const override
        {
            return _operator.sparsity_pattern();
        }

        bool matrix_is_spd() const override
        {
            return _operator.matrix_is_spd();
        }

        void assemble_matrix(Mat& A) const override
        {
            _operator.assemble_matrix(A);

            // A = I + _dt*A
            MatScale(A, _dt); // A = _dt*A;
            MatShift(A, 1);   // A = A + 1*I

            PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
            MatSetOption(A, MAT_SPD, is_spd);
        }

        void enforce_bc(Vec& b, const field_t& solution) const
        {
            _operator.enforce_bc(b, solution);
        }
        
    private:
        void assemble_scheme_on_uniform_grid(Mat&) const override {}
        void assemble_boundary_conditions(Mat&) const override {}
        void assemble_projection(Mat&) const override {}
        void assemble_prediction(Mat&) const override {}
    };
    
}} // end namespace