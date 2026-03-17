#pragma once
#include "matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Zero block
         */
        template <>
        class Assembly<int> : public MatrixAssembly
        {
          public:

            using scheme_t       = int;  // deactivate compatibility test in block_operator.tie_unknowns()
            using input_field_t  = void; // deactivate compatibility test during assembly
            using output_field_t = void;

          private:

            PetscInt m_owned_rows = 0;
            PetscInt m_owned_cols = 0;

            PetscInt m_local_rows = 0;
            PetscInt m_local_cols = 0;

          public:

            explicit Assembly(int value)
            {
                if (value != 0)
                {
                    std::cerr << "Unimplemented Assembly(" << value << ")" << std::endl;
                    exit(EXIT_FAILURE);
                }
                this->set_name("0");
            }

            void set_scheme(const scheme_t&)
            {
            }

            PetscInt owned_matrix_rows() const override
            {
                return m_owned_rows;
            }

            PetscInt owned_matrix_cols() const override
            {
                return m_owned_cols;
            }

            PetscInt local_matrix_rows() const override
            {
                return m_local_rows;
            }

            PetscInt local_matrix_cols() const override
            {
                return m_local_cols;
            }

            void set_owned_matrix_rows(PetscInt rows)
            {
                m_owned_rows = rows;
            }

            void set_owned_matrix_cols(PetscInt cols)
            {
                m_owned_cols = cols;
            }

            void set_local_matrix_rows(PetscInt rows)
            {
                m_local_rows = rows;
            }

            void set_local_matrix_cols(PetscInt cols)
            {
                m_local_cols = cols;
            }

            template <class Anything>
            void setup(const Anything&)
            {
            }

            const std::vector<PetscInt>& local_to_global_rows() const override
            {
                static std::vector<PetscInt> dummy;
                return dummy;
            }

            const std::vector<PetscInt>& local_to_global_cols() const override
            {
                static std::vector<PetscInt> dummy;
                return dummy;
            }

            void set_row_numbering(Numbering&)
            {
            }

            void set_col_numbering(Numbering&)
            {
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
            {
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
            {
            }

            void sparsity_pattern_projection(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
            {
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>&, std::vector<PetscInt>&) const override
            {
            }

            void assemble_scheme(Mat&) override
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

        using ZeroBlockAssembly = Assembly<int>;

    } // end namespace petsc
} // end namespace samurai
