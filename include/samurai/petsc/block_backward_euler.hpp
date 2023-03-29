#pragma once
#include "block_assembly.hpp"

namespace samurai 
{ 
    namespace petsc
    {
        template <int rows, int cols, class... Operators>
        class BlockBackwardEuler
        {
        public:
            static constexpr int n_rows = rows;
            static constexpr int n_cols = cols;
            using BlockOperator = BlockAssembly<rows, cols, Operators...>;
        private:
            BlockOperator& m_operator;
            double m_dt;

        public:
            BlockBackwardEuler(BlockOperator& op, double dt) : 
                m_operator(op),
                m_dt(dt)
            {}

            std::array<std::string, cols> field_names() const
            {
                return m_operator.field_names();
            }

            void create_matrix(Mat& A)
            {
                m_operator.create_matrix(A);
            }

            void assemble_matrix(Mat& A)
            {
                m_operator.assemble_matrix(A);
                
                MatScale(m_operator.block(0, 0), m_dt); // m_dt*Diff
                MatScale(m_operator.block(0, 1), m_dt); // m_dt*Grad
                MatShift(m_operator.block(0, 0), 1);    // I + m_dt*Diff
            }

            void enforce_bc(std::array<Vec, rows>& b) const
            {
                m_operator.enforce_bc(b);
            }

            void enforce_projection_prediction(std::array<Vec, rows>& b) const
            {
                m_operator.enforce_projection_prediction(b);
            }

            void add_0_for_useless_ghosts(std::array<Vec, rows>& b) const
            {
                m_operator.add_0_for_useless_ghosts(b);
            }

            std::array<Vec, cols> create_solution_vectors() const
            {
                return m_operator.create_solution_vectors();
            }

            template<class... Fields>
            auto tie(Fields&... fields) const
            {
                return m_operator.tie(fields...);
            }
        };

        

        template <int rows, int cols, class... Operators>
        auto make_backward_euler(BlockAssembly<rows, cols, Operators...>& op, double dt)
        {
            return BlockBackwardEuler<rows, cols, Operators...>(op, dt);
        }
        
    } // end namespace petsc
} // end namespace samurai