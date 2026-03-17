#pragma once
#include "global_numbering.hpp"
#include "matrix_assembly.hpp"
#include "utils.hpp"

namespace samurai
{
    namespace petsc
    {
        template <BlockAssemblyType assembly_type_, std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly;

        template <class OutputField, class InputField>
        class ManualAssembly : public MatrixAssembly
        {
          public:

            using scheme_t       = ManualAssembly<OutputField, InputField>;
            using input_field_t  = InputField;
            using output_field_t = OutputField;

          private:

            input_field_t* m_unknown = nullptr;

          protected:

            Numbering* m_row_numbering = nullptr;
            Numbering* m_col_numbering = nullptr;

          public:

            void set_scheme(const scheme_t&)
            {
            }

            input_field_t& unknown() const
            {
                return *m_unknown;
            }

            input_field_t* unknown_ptr() const
            {
                return m_unknown;
            }

            void set_unknown(input_field_t& unknown)
            {
                m_unknown = &unknown;
            }

            const std::vector<PetscInt>& local_to_global_rows() const override
            {
                assert(m_row_numbering != nullptr);
                return m_row_numbering->local_to_global_mapping;
            }

            const std::vector<PetscInt>& local_to_global_cols() const override
            {
                assert(m_col_numbering != nullptr);
                return m_col_numbering->local_to_global_mapping;
            }

            /**
             * This function is called in case of monolithic block_assembly
             */
            template <std::size_t rows_, std::size_t cols_, class... Operators>
            void setup(BlockAssembly<BlockAssemblyType::Monolithic, rows_, cols_, Operators...>& block_assembly)
            {
                m_row_numbering = &block_assembly.numbering();
                m_col_numbering = m_row_numbering;
            }

            /**
             * This function is called in case of nested block_assembly
             */
            template <std::size_t rows_, std::size_t cols_, class... Operators>
            void setup(BlockAssembly<BlockAssemblyType::NestedMatrices, rows_, cols_, Operators...>& /*block_assembly*/)
            {
                m_ghosts_row_shift = owned_matrix_rows();
                m_ghosts_col_shift = owned_matrix_cols();
            }

            Vec create_solution_vector(const input_field_t& field) const
            {
#ifdef SAMURAI_WITH_MPI
                Vec v = create_petsc_vector(owned_matrix_cols());
                copy_unknown(field, v);
#else
                Vec v = create_petsc_vector_from(field);
#endif
                return v;
            }

            virtual void copy_unknown(const input_field_t& field, Vec& x) const
            {
                if constexpr (is_xtensor<input_field_t>)
                {
#ifdef SAMURAI_WITH_MPI
                    if (mpi::communicator().size() > 1)
                    {
                        std::cerr << "This function is not implemented for MPI yet in ManualAssembly." << std::endl;
                        exit(EXIT_FAILURE);
                    }
#endif
                    copy(field, x, this->block_col_shift());
                }
                else
                {
                    // static_assert(std::is_same_v<UnknownField, void>, "");
                    std::cerr << "Please implement 'void copy_unknown(const UnknownField& field, Vec& x) const override' for your ManualAssembly subclass '"
                              << typeid(*this).name() << "'." << std::endl;
                }
            }

            void compute_block_numbering()
            {
                assert(false && "Not implemented yet");
            }

            /**
             * This function is called in case of stand-alone assembly (e.g., Poisson equation) or nested block_assembly (e.g., Stokes
             * equation).
             */
            void compute_numbering()
            {
                assert(false && "Not implemented yet");
            }

            void compute_local_to_global_rows(std::vector<PetscInt>&)
            {
                assert(false && "Not implemented yet");
            }

            void compute_local_to_global_rows()
            {
                compute_local_to_global_rows(m_row_numbering->local_to_global_mapping);
            }

            const auto& row_numbering() const
            {
                assert(m_row_numbering != nullptr);
                return *m_row_numbering;
            }

            auto& row_numbering()
            {
                assert(m_row_numbering != nullptr);
                return *m_row_numbering;
            }

            const auto& col_numbering() const
            {
                assert(m_col_numbering != nullptr);
                return *m_col_numbering;
            }

            auto& col_numbering()
            {
                assert(m_col_numbering != nullptr);
                return *m_col_numbering;
            }

            void set_row_numbering(Numbering& numbering)
            {
                m_row_numbering = &numbering;
            }

            void set_col_numbering(Numbering& numbering)
            {
                m_col_numbering = &numbering;
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
