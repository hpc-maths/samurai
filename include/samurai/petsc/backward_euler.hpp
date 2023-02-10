#pragma once
#include "matrix_assembly.hpp"

namespace samurai { namespace petsc
{
    /**
     * @class BackwardEuler
    */
    template<class Operator>
    class BackwardEuler : public MatrixAssembly
    {
    public:
        using field_t = typename Operator::field_t;
        using Mesh = typename field_t::mesh_t;
    private:
        Operator& _operator;
        double _dt;

    public:
        BackwardEuler(Operator& op, double dt) : 
            _operator(op),
            _dt(dt)
        {}

        auto& unknown() const
        {
            return _operator.unknown();
        }

        auto& mesh() const
        {
            return _operator.mesh();
        }

        PetscInt matrix_rows() const override
        {
            return _operator.matrix_rows();
        }

        PetscInt matrix_cols() const override
        {
            return _operator.matrix_cols();
        }

        std::vector<PetscInt> sparsity_pattern() const override
        {
            return _operator.sparsity_pattern();
        }

        bool matrix_is_spd() const override
        {
            return _operator.matrix_is_spd();
        }

        void assemble_matrix(Mat& A) override
        {
            _operator.assemble_matrix(A);

            // A = I + _dt*A
            MatScale(A, _dt); // A = _dt*A;
            MatShift(A, 1);   // A = A + 1*I

            PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
            MatSetOption(A, MAT_SPD, is_spd);
        }

        void enforce_bc(Vec& b) const
        {
            _operator.enforce_bc(b);
        }
        
    private:
        void assemble_scheme_on_uniform_grid(Mat&) override {}
        void assemble_boundary_conditions(Mat&) override {}
        void assemble_projection(Mat&) const override {}
        void assemble_prediction(Mat&) const override {}
    };

    

    template<class Operator>
    auto make_backward_euler(Operator& op, double dt)
    {
        return BackwardEuler<Operator>(op, dt);
    }
    
}} // end namespace