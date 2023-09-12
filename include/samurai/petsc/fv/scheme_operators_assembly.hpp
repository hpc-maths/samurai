#pragma once
#include "../../schemes/fv/scheme_operators.hpp"
#include "../matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {

        /**
         * Addition of a flux-based scheme and a cell-based scheme.
         * The cell-based scheme is assembled first, then the flux-based scheme.
         * The boundary conditions are taken from the flux-based scheme.
         */
        template <class FluxScheme, class CellScheme>
        class Assembly<FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>> : public MatrixAssembly
        {
          public:

            using scheme_t = FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>;
            using field_t  = typename FluxScheme::field_t;

          private:

            const scheme_t* m_sum_scheme;
            field_t* m_unknown = nullptr;

            Assembly<FluxScheme> m_flux_assembly;
            Assembly<CellScheme> m_cell_assembly;

          public:

            explicit Assembly(const scheme_t& sum_scheme)
                : m_sum_scheme(&sum_scheme)
                , m_flux_assembly(sum_scheme.flux_scheme())
                , m_cell_assembly(sum_scheme.cell_scheme())
            {
                this->set_name(sum_scheme.name());
            }

            auto& unknown() const
            {
                return m_flux_assembly.unknown();
            }

            auto unknown_ptr() const
            {
                return m_flux_assembly.unknown_ptr();
            }

            bool undefined_unknown() const
            {
                return !m_flux_assembly.unknown_ptr();
            }

            void set_unknown(field_t& unknown)
            {
                m_flux_assembly.set_unknown(unknown);
                m_cell_assembly.set_unknown(unknown);
            }

            auto& scheme() const
            {
                return *m_sum_scheme;
            }

            InsertMode current_insert_mode() const
            {
                return m_flux_assembly.current_insert_mode();
            }

            void set_current_insert_mode(InsertMode insert_mode)
            {
                m_flux_assembly.set_current_insert_mode(insert_mode);
                m_cell_assembly.set_current_insert_mode(insert_mode);
            }

            void set_is_block(bool is_block) override
            {
                MatrixAssembly::set_is_block(is_block);
                m_flux_assembly.set_is_block(is_block);
                m_cell_assembly.set_is_block(is_block);
            }

            PetscInt matrix_rows() const override
            {
                if (m_flux_assembly.matrix_rows() != m_cell_assembly.matrix_rows())
                {
                    std::cerr << "Invalid '+' operation: both schemes must generate the same number of matrix rows." << std::endl;
                    std::cerr << "                       '" << m_flux_assembly.name() << "': " << m_flux_assembly.matrix_rows() << ", "
                              << m_cell_assembly.name() << ": " << m_cell_assembly.matrix_rows() << std::endl;
                    assert(false);
                }
                return m_flux_assembly.matrix_rows();
            }

            PetscInt matrix_cols() const override
            {
                if (m_flux_assembly.matrix_cols() != m_cell_assembly.matrix_cols())
                {
                    std::cerr << "Invalid '+' operation: both schemes must generate the same number of matrix columns." << std::endl;
                    std::cerr << "                       '" << m_flux_assembly.name() << "': " << m_flux_assembly.matrix_cols() << ", "
                              << m_cell_assembly.name() << ": " << m_cell_assembly.matrix_cols() << std::endl;
                    assert(false);
                }
                return m_flux_assembly.matrix_cols();
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                // To be safe, allocate for both schemes (nnz is the sum of both)
                m_cell_assembly.sparsity_pattern_scheme(nnz);
                m_flux_assembly.sparsity_pattern_scheme(nnz);
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // Only the flux scheme will assemble the boundary conditions
                m_flux_assembly.sparsity_pattern_boundary(nnz);
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                m_flux_assembly.sparsity_pattern_projection(nnz);
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                m_flux_assembly.sparsity_pattern_prediction(nnz);
            }

            void assemble_scheme(Mat& A) override
            {
                // First the cell-based scheme because it uses INSERT_VALUES
                m_cell_assembly.assemble_scheme(A);
                // Then the flux-based scheme
                m_flux_assembly.set_current_insert_mode(m_cell_assembly.current_insert_mode());
                m_flux_assembly.assemble_scheme(A);
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the boundary conditions in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_assembly.assemble_boundary_conditions(A);
            }

            void assemble_projection(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the projection operator in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_assembly.assemble_projection(A);
            }

            void assemble_prediction(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the prediction operator in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_assembly.assemble_prediction(A);
            }

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                m_flux_assembly.add_1_on_diag_for_useless_ghosts(A);
            }

            void enforce_bc(Vec& b) const
            {
                m_flux_assembly.enforce_bc(b);
            }

            void add_0_for_useless_ghosts(Vec& b) const
            {
                m_flux_assembly.add_0_for_useless_ghosts(b);
            }

            void enforce_projection_prediction(Vec& b) const
            {
                m_flux_assembly.enforce_projection_prediction(b);
            }

            bool matrix_is_symmetric() const override
            {
                return m_flux_assembly.matrix_is_symmetric() && m_cell_assembly.matrix_is_symmetric();
            }

            bool matrix_is_spd() const override
            {
                return m_flux_assembly.matrix_is_spd() && m_cell_assembly.matrix_is_spd();
            }

            void reset() override
            {
                m_flux_assembly.reset();
                m_cell_assembly.reset();
            }
        };

    } // end namespace petsc
} // end namespace samurai
