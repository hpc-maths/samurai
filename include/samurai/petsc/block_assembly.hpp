#pragma once
#include "matrix_assembly.hpp"
#include "utils.hpp"

namespace samurai 
{ 
    namespace petsc
    {
        template <int rows, int cols, class... Operators>
        class BlockAssembly
        {
        private:
            std::tuple<Operators&...> m_operators;
            std::array<Mat, rows * cols> m_blocks;
        public:
            BlockAssembly(Operators&... operators) :
                m_operators(operators...)
            {
                static constexpr std::size_t n_operators = sizeof...(operators);
                static_assert(n_operators == rows * cols, "The number of operators must correspond to rows*cols.");

                std::size_t i = 0;
                for_each(m_operators, [&](auto& op)
                {
                    auto row = i / cols;
                    auto col = i % cols;
                    m_blocks[i] = nullptr;
                    bool diagonal_block = (row == col);
                    op.add_1_on_diag_for_useless_ghosts_if(diagonal_block);
                    op.include_bc_if(diagonal_block);
                    op.assemble_proj_pred_if(diagonal_block);
                    i++;
                });
            }

            std::array<std::string, cols> field_names() const
            {
                std::array<std::string, cols> names;
                std::size_t i = 0;
                for_each(m_operators, [&](auto& op)
                {
                    auto row = i / cols;
                    auto col = i % cols;
                    if (row == col)
                    {
                        names[col] = op.unknown().name();
                    }
                    i++;
                });
                return names;
            }

            void create_matrix(Mat& A)
            {
                std::size_t i = 0;
                for_each(m_operators, [&](auto& op)
                {
                    /*auto row = i / cols;
                    auto col = i % cols;
                    std::cout << "create_matrix (" << row << ", " << col << ")" << std::endl;*/
                    op.create_matrix(m_blocks[i]);
                    i++;
                });

                MatCreateNest(PETSC_COMM_SELF, rows, PETSC_NULL, cols, PETSC_NULL, m_blocks.data(), &A);
            }

            void assemble_matrix(Mat& A)
            {
                std::size_t i = 0;
                for_each(m_operators, [&](auto& op)
                {
                    /*auto row = i / cols;
                    auto col = i % cols;
                    std::cout << "assemble_matrix (" << row << ", " << col << ")" << std::endl;*/
                    op.assemble_matrix(m_blocks[i]);
                    i++;
                });
                MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
            }

            void enforce_bc(std::array<Vec, rows>& b) const
            {
                std::size_t i = 0;
                for_each(m_operators, [&](const auto& op) 
                {
                    auto row = i / cols;
                    //auto col = i % cols;
                    if (op.include_bc())
                    {
                        //std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                        op.enforce_bc(b[row]);
                    }
                    i++;
                });
            }

            void enforce_projection_prediction(std::array<Vec, rows>& b) const
            {
                std::size_t i = 0;
                for_each(m_operators, [&](const auto& op) 
                {
                    auto row = i / cols;
                    //auto col = i % cols;
                    if (op.assemble_proj_pred())
                    {
                        //std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                        op.enforce_projection_prediction(b[row]);
                    }
                    i++;
                });
            }

            std::array<Vec, cols> create_solution_vectors() const
            {
                std::array<Vec, cols> x_blocks;
                std::size_t i = 0;
                for_each(m_operators, [&](const auto& op) 
                {
                    auto row = i / cols;
                    auto col = i % cols;
                    if (row == 0)
                    {
                        x_blocks[col] = create_petsc_vector_from(op.unknown());
                        PetscObjectSetName(reinterpret_cast<PetscObject>(x_blocks[col]), op.unknown().name().c_str());
                    }
                    i++;
                });
                return x_blocks;
            }

            template<class... Fields>
            auto tie(Fields&... fields) const
            {
                static constexpr std::size_t n_fields = sizeof...(fields);
                static_assert(n_fields == rows, "The number of fields must correspond to the number of rows of the block operator.");

                return std::tuple<Fields&...>(fields...);
            }

        };

        template <int rows, int cols, class... Operators>
        auto make_block_operator(Operators&... operators)
        {
            return BlockAssembly<rows, cols, Operators...>(operators...);
        }
        
    } // end namespace petsc
} // end namespace samurai