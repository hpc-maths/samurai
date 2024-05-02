#pragma once
#include "simple_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class Field>
        struct ZeroBlock : public SimpleAssembly<Field>
        {
            ZeroBlock()
            {
                this->set_fit_block_dimensions(true);
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
