#pragma once
#include "simple_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class Field>
        struct ZeroBlock : public SimpleAssembly<Field>
        {
            std::size_t rows;
            std::size_t cols;

            PetscInt matrix_rows() const override
            {
                return static_cast<PetscInt>(rows);
            }

            PetscInt matrix_cols() const override
            {
                return static_cast<PetscInt>(cols);
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>&) const override
            {
            }

            void assemble_scheme(Mat&) override
            {
            }
        };

    } // end namespace petsc
} // end namespace samurai
