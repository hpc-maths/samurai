#pragma once
#include "../print.hpp"
#include "matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Zero block
         */
        template <>
        struct Assembly<int> : public MatrixAssembly
        {
            using scheme_t = int;  // deactivate compatibility test in block_operator.tie_unknowns()
            using field_t  = void; // deactivate compatibility test during assembly

            explicit Assembly(int value)
            {
                if (value != 0)
                {
                    samurai::io::eprint("Unimplemented Assembly({})\n", value);
                    exit(EXIT_FAILURE);
                }
                this->fit_block_dimensions(true);
                this->set_name("0");
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

            void insert_value_on_diag_for_useless_ghosts(Mat&) override
            {
            }

            template <class Func>
            void for_each_useless_ghost_row(Func&&) const
            {
            }
        };

    } // end namespace petsc
} // end namespace samurai
