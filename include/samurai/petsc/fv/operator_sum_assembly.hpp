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

            using scheme_t       = OperatorSum<Operators...>;
            using input_field_t  = typename scheme_t::field_t;
            using output_field_t = typename scheme_t::output_field_t;
            using cell_t         = typename input_field_t::mesh_t::cell_t;

          private:

            scheme_t m_sum_scheme;

            std::tuple<Assembly<Operators>...> m_assembly_ops;

          public:

            explicit Assembly(const scheme_t& sum_scheme)
                : m_sum_scheme(sum_scheme)
                , m_assembly_ops(transform(m_sum_scheme.operators(),
                                           [](auto& op)
                                           {
                                               return make_assembly(op);
                                           }))
            {
                this->set_name(sum_scheme.name());
            }

            void set_scheme(const scheme_t& s)
            {
                m_sum_scheme = s;
                static_for<0, sizeof...(Operators)>::apply(
                    [&](auto i)
                    {
                        std::get<i>(m_assembly_ops).set_scheme(std::get<i>(s.operators()));
                    });
            }

            constexpr const auto& largest_stencil_assembly() const
            {
                return std::get<scheme_t::largest_stencil_index>(m_assembly_ops);
            }

            constexpr auto& largest_stencil_assembly()
            {
                return std::get<scheme_t::largest_stencil_index>(m_assembly_ops);
            }

            auto& unknown() const
            {
                return largest_stencil_assembly().unknown();
            }

            auto unknown_ptr() const
            {
                return largest_stencil_assembly().unknown_ptr();
            }

            bool undefined_unknown() const
            {
                return !unknown_ptr();
            }

            const auto& mesh() const // cppcheck-suppress duplInheritedMember
            {
                return unknown().mesh();
            }

            auto& mesh() // cppcheck-suppress duplInheritedMember
            {
                return unknown().mesh();
            }

            void set_unknown(input_field_t& unknown)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_unknown(unknown);
                         });
            }

            auto& scheme() // cppcheck-suppress duplInheritedMember
            {
                return m_sum_scheme;
            }

            auto& scheme() const
            {
                return m_sum_scheme;
            }

            const auto& row_numbering() const // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().row_numbering();
            }

            auto& row_numbering() // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().row_numbering();
            }

            const auto& col_numbering() const // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().col_numbering();
            }

            auto& col_numbering() // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().col_numbering();
            }

            void set_row_numbering(Numbering& numbering) // cppcheck-suppress duplInheritedMember
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_row_numbering(numbering);
                         });
            }

            void set_col_numbering(const Numbering& numbering) // cppcheck-suppress duplInheritedMember
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_col_numbering(numbering);
                         });
            }

            void set_block_row_shift(PetscInt shift) override
            {
                MatrixAssembly::set_block_row_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_block_row_shift(shift);
                         });
            }

            void set_block_col_shift(PetscInt shift) override
            {
                MatrixAssembly::set_block_col_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_block_col_shift(shift);
                         });
            }

            void set_rank_row_shift(PetscInt shift) override
            {
                MatrixAssembly::set_rank_row_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_rank_row_shift(shift);
                         });
            }

            void set_rank_col_shift(PetscInt shift) override
            {
                MatrixAssembly::set_rank_col_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_rank_col_shift(shift);
                         });
            }

            void set_ghosts_row_shift(PetscInt shift) override
            {
                MatrixAssembly::set_ghosts_row_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_ghosts_row_shift(shift);
                         });
            }

            void set_ghosts_col_shift(PetscInt shift) override
            {
                MatrixAssembly::set_ghosts_col_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_ghosts_col_shift(shift);
                         });
            }

            void set_current_insert_mode(InsertMode insert_mode) override
            {
                MatrixAssembly::set_current_insert_mode(insert_mode);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_current_insert_mode(insert_mode);
                         });
            }

            void is_block_in_monolithic_matrix(bool is_block) override
            {
                MatrixAssembly::is_block_in_monolithic_matrix(is_block);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.is_block_in_monolithic_matrix(is_block);
                         });
            }

            void is_block_in_nested_matrix(bool is_block) override
            {
                MatrixAssembly::is_block_in_nested_matrix(is_block);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.is_block_in_nested_matrix(is_block);
                         });
            }

            PetscInt owned_matrix_rows() const override
            {
                auto rows = std::get<0>(m_assembly_ops).owned_matrix_rows();
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (op.owned_matrix_rows() != rows)
                             {
                                 std::cerr << "Invalid '+' operation: all schemes must generate the same number of matrix rows." << std::endl;
                                 std::cerr << "                       '" << std::get<0>(m_assembly_ops).name()
                                           << "': " << std::get<0>(m_assembly_ops).owned_matrix_rows() << ", " << op.name() << ": "
                                           << op.owned_matrix_rows() << std::endl;
                                 assert(false);
                             }
                         });
                return rows;
            }

            PetscInt owned_matrix_cols() const override
            {
                auto cols = std::get<0>(m_assembly_ops).owned_matrix_cols();
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (op.owned_matrix_cols() != cols)
                             {
                                 std::cerr << "Invalid '+' operation: all schemes must generate the same number of matrix columns."
                                           << std::endl;
                                 std::cerr << "                       '" << std::get<0>(m_assembly_ops).name()
                                           << "': " << std::get<0>(m_assembly_ops).owned_matrix_cols() << ", " << op.name() << ": "
                                           << op.owned_matrix_cols() << std::endl;
                                 assert(false);
                             }
                         });
                return cols;
            }

            PetscInt local_matrix_rows() const override
            {
                return largest_stencil_assembly().local_matrix_rows();
            }

            PetscInt local_matrix_cols() const override
            {
                return largest_stencil_assembly().local_matrix_cols();
            }

            Vec create_rhs_vector(const output_field_t& field) const // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().create_rhs_vector(field);
            }

            void copy_rhs(const output_field_t& field, Vec& v) const // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().copy_rhs(field, v);
            }

            void copy_rhs(const Vec& v, output_field_t& field) const // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().copy_rhs(v, field);
            }

            Vec create_solution_vector(const input_field_t& field) const // cppcheck-suppress duplInheritedMember
            {
                return largest_stencil_assembly().create_solution_vector(field);
            }

            void copy_unknown(const Vec& v, input_field_t& field) const // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().copy_unknown(v, field);
            }

            void copy_unknown(const input_field_t& field, Vec& v) const // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().copy_unknown(field, v);
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                // The scheme with largest stencil allocates the number of non-zeros.
                largest_stencil_assembly().sparsity_pattern_scheme(d_nnz, o_nnz);
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                // Only one scheme assembles the boundary conditions.
                // We arbitrarily choose the one with largest stencil,
                // because we already use it to allocate the number of non-zeros in the scheme.
                largest_stencil_assembly().sparsity_pattern_boundary(d_nnz, o_nnz);
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                largest_stencil_assembly().sparsity_pattern_projection(d_nnz, o_nnz);
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                largest_stencil_assembly().sparsity_pattern_prediction(d_nnz, o_nnz);
            }

            void assemble_scheme(Mat& A) override
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             // std::cout << "Assembly of " << op.name() << std::endl;
                             op.assemble_scheme(A);
                             set_current_insert_mode(op.current_insert_mode());
                         });
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                // We hope that all schemes implement the boundary conditions in the same fashion,
                // and arbitrarily choose the one with largest stencil
                // (because this is the one used to allocate the number of non-zeros in the scheme,
                // so we use it everywhere).
                largest_stencil_assembly().assemble_boundary_conditions(A);
            }

            void assemble_projection(Mat& A) override
            {
                // We hope that all schemes implement the projection operator in the same fashion,
                // and arbitrarily choose the one with largest stencil.
                largest_stencil_assembly().assemble_projection(A);
            }

            void assemble_prediction(Mat& A) override
            {
                // We hope that all schemes implement the prediction operator in the same fashion,
                // and arbitrarily choose the one with largest stencil.
                largest_stencil_assembly().assemble_prediction(A);
            }

            void include_boundary_fluxes(bool include)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             using op_scheme_t = typename std::decay_t<decltype(op)>::scheme_t;

                             if constexpr (is_FluxBasedScheme_v<op_scheme_t>)
                             {
                                 op.include_boundary_fluxes(include);
                             }
                         });
            }

            void set_diag_value_for_useless_ghosts(PetscScalar value) override
            {
                MatrixAssembly::set_diag_value_for_useless_ghosts(value);
                largest_stencil_assembly().set_diag_value_for_useless_ghosts(value);
            }

            void insert_value_on_diag_for_useless_ghosts(Mat& A) override
            {
                largest_stencil_assembly().insert_value_on_diag_for_useless_ghosts(A);
            }

            template <class Func>
            void for_each_useless_ghost_row(Func&& f) const // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().for_each_useless_ghost_row(std::forward<Func>(f));
            }

            void set_0_for_all_ghosts(Vec& b) const
            {
                largest_stencil_assembly().set_0_for_all_ghosts(b);
            }

            void enforce_bc(Vec& b) const
            {
                largest_stencil_assembly().enforce_bc(b);
            }

            void set_0_for_useless_ghosts(Vec& b) const
            {
                largest_stencil_assembly().set_0_for_useless_ghosts(b);
            }

            void enforce_projection_prediction(Vec& b) const
            {
                largest_stencil_assembly().enforce_projection_prediction(b);
            }

            SAMURAI_INLINE PetscInt col_index(PetscInt cell_index, unsigned int field_j) const
            {
                return largest_stencil_assembly().col_index(cell_index, field_j);
            }

            SAMURAI_INLINE PetscInt row_index(PetscInt cell_index, [[maybe_unused]] unsigned int field_i) const
            {
                return largest_stencil_assembly().row_index(cell_index, field_i);
            }

            SAMURAI_INLINE PetscInt col_index(const cell_t& cell, unsigned int field_j) const
            {
                return largest_stencil_assembly().col_index(cell, field_j);
            }

            SAMURAI_INLINE PetscInt row_index(const cell_t& cell, unsigned int field_i) const
            {
                return largest_stencil_assembly().row_index(cell, field_i);
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

            void setup() override
            {
                // computes global numbering and other costly information
                largest_stencil_assembly().setup();

                std::size_t i = 0;
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (i != scheme_t::largest_stencil_index)
                             {
                                 // set up the other schemes by giving the all the costly information to avoid recomputing it
                                 op.setup(largest_stencil_assembly());
                             }
                             ++i;
                         });
            }

            /**
             * This function is called in case of block_assembly.
             */
            template <BlockAssemblyType assembly_type_, std::size_t rows_, std::size_t cols_, class... Operators_>
            void setup(BlockAssembly<assembly_type_, rows_, cols_, Operators_...>& block_assembly)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             // set up the other schemes by giving the all the costly information to avoid recomputing it
                             op.setup(block_assembly);
                         });
            }

            void compute_numbering() // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().compute_numbering();

                std::size_t i = 0;
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (i != scheme_t::largest_stencil_index)
                             {
                                 op.set_row_numbering(largest_stencil_assembly().row_numbering());
                                 op.set_col_numbering(largest_stencil_assembly().col_numbering());
                             }
                             ++i;
                         });
            }

            void compute_block_numbering() // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().compute_block_numbering();
            }

            const std::vector<PetscInt>& local_to_global_rows() const override
            {
                return largest_stencil_assembly().local_to_global_rows();
            }

            const std::vector<PetscInt>& local_to_global_cols() const override
            {
                return largest_stencil_assembly().local_to_global_cols();
            }

            void compute_local_to_global_rows(std::vector<PetscInt>& local_to_global_rows) // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().compute_local_to_global_rows(local_to_global_rows);
            }

            void compute_local_to_global_rows() // cppcheck-suppress duplInheritedMember
            {
                largest_stencil_assembly().compute_local_to_global_rows();
            }
        };

    } // end namespace petsc
} // end namespace samurai
