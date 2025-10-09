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

            void set_unknown(input_field_t& unknown)
            {
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_unknown(unknown);
                         });
            }

            auto& scheme() // cppcheck-suppress functionRedefined
            {
                return m_sum_scheme;
            }

            auto& scheme() const
            {
                return m_sum_scheme;
            }

            void set_row_shift(PetscInt shift) override
            {
                MatrixAssembly::set_row_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_row_shift(shift);
                         });
            }

            void set_col_shift(PetscInt shift) override
            {
                MatrixAssembly::set_col_shift(shift);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.set_col_shift(shift);
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

            void is_block(bool is_block) override
            {
                MatrixAssembly::is_block(is_block);

                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             op.is_block(is_block);
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

            Vec create_rhs_vector(const output_field_t& field) const
            {
                return largest_stencil_assembly().create_rhs_vector(field);
            }

            void copy_rhs(const output_field_t& field, Vec& v) const
            {
                largest_stencil_assembly().copy_rhs(field, v);
            }

            Vec create_solution_vector(const input_field_t& field) const
            {
                return largest_stencil_assembly().create_solution_vector(field);
            }

            void update_unknown(Vec& v) const
            {
                largest_stencil_assembly().update_unknown(v);
            }

#ifdef SAMURAI_WITH_MPI
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
#else
            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                // The scheme with largest stencil allocates the number of non-zeros.
                largest_stencil_assembly().sparsity_pattern_scheme(nnz);
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // Only one scheme assembles the boundary conditions.
                // We arbitrarily choose the one with largest stencil,
                // because we already use it to allocate the number of non-zeros in the scheme.
                largest_stencil_assembly().sparsity_pattern_boundary(nnz);
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                largest_stencil_assembly().sparsity_pattern_projection(nnz);
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                largest_stencil_assembly().sparsity_pattern_prediction(nnz);
            }
#endif

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

            inline PetscInt col_index(PetscInt cell_index, unsigned int field_j) const
            {
                return largest_stencil_assembly().col_index(cell_index, field_j);
            }

            inline PetscInt row_index(PetscInt cell_index, [[maybe_unused]] unsigned int field_i) const
            {
                return largest_stencil_assembly().row_index(cell_index, field_i);
            }

            inline PetscInt col_index(const cell_t& cell, unsigned int field_j) const
            {
                return largest_stencil_assembly().col_index(cell, field_j);
            }

            inline PetscInt row_index(const cell_t& cell, unsigned int field_i) const
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

            void reset() override
            {
                // computes global numbering and other costly information
                largest_stencil_assembly().reset();

                std::size_t i = 0;
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             if (i != scheme_t::largest_stencil_index)
                             {
                                 // reset the other schemes by giving the all the costly information to avoid recomputing it
                                 op.reset(largest_stencil_assembly());
                             }
                             ++i;
                         });
            }

            void set_local_to_global_mappings(Mat& A) const override
            {
                largest_stencil_assembly().set_local_to_global_mappings(A);
            }
        };

    } // end namespace petsc
} // end namespace samurai
