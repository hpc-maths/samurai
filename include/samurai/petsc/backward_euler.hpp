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
        Operator& m_operator;
        double m_dt;

    public:
        BackwardEuler(Operator& op, double dt) : 
            m_operator(op),
            m_dt(dt)
        {}

        auto& unknown() const
        {
            return m_operator.unknown();
        }

        auto& mesh() const
        {
            return m_operator.mesh();
        }

        PetscInt matrix_rows() const override
        {
            return m_operator.matrix_rows();
        }

        PetscInt matrix_cols() const override
        {
            return m_operator.matrix_cols();
        }

        std::vector<PetscInt> sparsity_pattern() const override
        {
            return m_operator.sparsity_pattern();
        }

        bool matrix_is_spd() const override
        {
            return m_operator.matrix_is_spd();
        }

        void assemble_matrix(Mat& A) override
        {
            m_operator.assemble_matrix(A);

            // A = I + m_dt*A
            MatScale(A, m_dt); // A = m_dt*A;
            MatShift(A, 1);   // A = A + 1*I

            PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
            MatSetOption(A, MAT_SPD, is_spd);
        }

        void enforce_bc(Vec& b) const
        {
            m_operator.enforce_bc(b);
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