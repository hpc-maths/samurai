#pragma once
#include "matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {

        template <class UnknownField>
        class ManualAssembly : public MatrixAssembly
        {
          public:

            using scheme_t = ManualAssembly<UnknownField>;
            using field_t  = UnknownField;

          private:

            UnknownField* m_unknown = nullptr;

          public:

            UnknownField& unknown() const
            {
                return *m_unknown;
            }

            void set_unknown(UnknownField& unknown)
            {
                m_unknown = &unknown;
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>&) const override
            {
            }

            void sparsity_pattern_projection(std::vector<PetscInt>&) const override
            {
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>&) const override
            {
            }

            void assemble_boundary_conditions(Mat&) override
            {
            }

            void assemble_projection(Mat&) override
            {
            }

            void assemble_prediction(Mat&) override
            {
            }

            void insert_value_on_diag_for_useless_ghosts(Mat&) override
            {
            }
        };

    } // end namespace petsc
} // end namespace samurai
