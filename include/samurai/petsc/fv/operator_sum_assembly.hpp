#pragma once
#include "../../schemes/fv/scheme_operators.hpp"
#include "../matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <class... Operators>
        class Assembly<OperatorSum<Operators...>> : public MatrixAssembly
        {
          public:

            using scheme_t = OperatorSum<Operators...>;
            using field_t  = typename scheme_t::field_t;

          private:

            const scheme_t* m_sum_scheme;
            field_t* m_unknown = nullptr;

            std::tuple<Assembly<Operators>...> m_assembly_ops;

          public:

            explicit Assembly(const scheme_t& sum_scheme)
                : m_sum_scheme(&sum_scheme)
                , m_assembly_ops(transform(sum_scheme.operators(),
                                           [](const auto& op)
                                           {
                                               return make_assembly(op);
                                           }))
            {
                this->set_name(sum_scheme.name());
            }

            auto& unknown() const
            {
                return std::get<0>(m_assembly_ops).unknown();
            }

            auto unknown_ptr() const
            {
                return std::get<0>(m_assembly_ops).unknown_ptr();
            }

            bool undefined_unknown() const
            {
                return !unknown_ptr();
            }

            void set_unknown(field_t& unknown)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_unknown(unknown);
                         });
            }

            auto& scheme() const
            {
                return *m_sum_scheme;
            }

            InsertMode current_insert_mode() const
            {
                return std::get<0>(m_assembly_ops).current_insert_mode();
            }

            void set_current_insert_mode(InsertMode insert_mode)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_current_insert_mode(insert_mode);
                         });
            }

            void set_is_block(bool is_block) override
            {
                MatrixAssembly::set_is_block(is_block);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_is_block(is_block);
                         });
            }

            PetscInt matrix_rows() const override
            {
                auto rows = std::get<0>(m_assembly_ops).matrix_rows();
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (op.matrix_rows() != rows)
                             {
                                 std::cerr << "Invalid '+' operation: all schemes must generate the same number of matrix rows." << std::endl;
                                 std::cerr << "                       '" << std::get<0>(m_assembly_ops).name()
                                           << "': " << std::get<0>(m_assembly_ops).matrix_rows() << ", " << op.name() << ": "
                                           << op.matrix_rows() << std::endl;
                                 assert(false);
                             }
                         });
                return rows;
            }

            PetscInt matrix_cols() const override
            {
                auto cols = std::get<0>(m_assembly_ops).matrix_cols();
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (op.matrix_cols() != cols)
                             {
                                 std::cerr << "Invalid '+' operation: all schemes must generate the same number of matrix columns."
                                           << std::endl;
                                 std::cerr << "                       '" << std::get<0>(m_assembly_ops).name()
                                           << "': " << std::get<0>(m_assembly_ops).matrix_cols() << ", " << op.name() << ": "
                                           << op.matrix_cols() << std::endl;
                                 assert(false);
                             }
                         });
                return cols;
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                // To be safe, allocate for all schemes (nnz is the sum)
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.sparsity_pattern_scheme(nnz);
                         });
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // Only one scheme will assemble the boundary conditions
                std::get<0>(m_assembly_ops).sparsity_pattern_boundary(nnz);
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                std::get<0>(m_assembly_ops).sparsity_pattern_projection(nnz);
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                std::get<0>(m_assembly_ops).sparsity_pattern_prediction(nnz);
            }

            void assemble_scheme(Mat& A) override
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.assemble_scheme(A);
                             set_current_insert_mode(op.current_insert_mode());
                         });
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                // We hope that all schemes implement the boundary conditions in the same fashion,
                // and arbitrarily choose the first one.
                std::get<0>(m_assembly_ops).assemble_boundary_conditions(A);
            }

            void assemble_projection(Mat& A) override
            {
                // We hope that all schemes implement the projection operator in the same fashion,
                // and arbitrarily choose the first one.
                std::get<0>(m_assembly_ops).assemble_projection(A);
            }

            void assemble_prediction(Mat& A) override
            {
                // We hope that all schemes implement the prediction operator in the same fashion,
                // and arbitrarily choose the first one.
                std::get<0>(m_assembly_ops).assemble_prediction(A);
            }

            void set_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                std::get<0>(m_assembly_ops).set_1_on_diag_for_useless_ghosts(A);
            }

            void set_0_for_all_ghosts(Vec& b) const
            {
                std::get<0>(m_assembly_ops).set_0_for_all_ghosts(b);
            }

            void enforce_bc(Vec& b) const
            {
                std::get<0>(m_assembly_ops).enforce_bc(b);
            }

            void set_0_for_useless_ghosts(Vec& b) const
            {
                std::get<0>(m_assembly_ops).set_0_for_useless_ghosts(b);
            }

            void enforce_projection_prediction(Vec& b) const
            {
                std::get<0>(m_assembly_ops).enforce_projection_prediction(b);
            }

            bool matrix_is_symmetric() const override
            {
                bool is_symmetric = true;
                for_each(m_assembly_ops,
                         [&](const auto& op)
                         {
                             is_symmetric = is_symmetric && op.matrix_is_symmetric();
                         });
                return is_symmetric;
            }

            bool matrix_is_spd() const override
            {
                bool is_spd = true;
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             is_spd = is_spd && op.matrix_is_spd();
                         });
                return is_spd;
            }

            void reset() override
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.reset();
                         });
            }
        };

    } // end namespace petsc
} // end namespace samurai
