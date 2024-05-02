#pragma once
#include "simple_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        // template <class Field>
        struct ZeroBlock : public MatrixAssembly //: public SimpleAssembly<Field>
        {
            using scheme_t = void; // deactivate compatibility test during assembly
            using field_t  = void; // deactivate compatibility test during assembly

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

            void set_1_on_diag_for_useless_ghosts(Mat&) override
            {
            }
        };

    } // end namespace petsc
} // end namespace samurai
