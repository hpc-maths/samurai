#pragma once

// #define ENABLE_MG
#include "block_assembly.hpp"
#include "fv/cell_based_scheme_assembly.hpp"
#include "fv/flux_based_scheme_assembly.hpp"
#include "fv/scheme_operators_assembly.hpp"
#ifdef ENABLE_MG
#include "multigrid/petsc/GeometricMultigrid.hpp"
#else
#include "utils.hpp"
#endif

namespace samurai
{
    namespace petsc
    {
        template <class Assembly>
        class SolverBase
        {
            using scheme_t = typename Assembly::scheme_t;

          protected:

            Assembly m_assembly;
            KSP m_ksp        = nullptr;
            Mat m_A          = nullptr;
            bool m_is_set_up = false;

          public:

            explicit SolverBase(const scheme_t& scheme)
                : m_assembly(scheme)
            {
                configure_default_solver();
            }

            virtual ~SolverBase()
            {
                destroy_petsc_objects();
            }

            virtual void destroy_petsc_objects()
            {
                if (m_A)
                {
                    MatDestroy(&m_A);
                    m_A = nullptr;
                }
                if (m_ksp)
                {
                    KSPDestroy(&m_ksp);
                    m_ksp = nullptr;
                }
            }

            SolverBase& operator=(const SolverBase& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_ksp       = other.m_ksp;
                    this->m_A         = other.m_A;
                    this->m_is_set_up = other.m_is_set_up;
                }
                return *this;
            }

            SolverBase& operator=(SolverBase&& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_ksp       = other.m_ksp;
                    this->m_A         = other.m_A;
                    this->m_is_set_up = other.m_is_set_up;
                    other.m_ksp       = nullptr; // Prevent KSP destruction when 'other' object is destroyed
                    other.m_A         = nullptr;
                    other.m_is_set_up = false;
                }
                return *this;
            }

            KSP& Ksp()
            {
                return m_ksp;
            }

            bool is_set_up()
            {
                return m_is_set_up;
            }

            auto& assembly()
            {
                return m_assembly;
            }

          private:

            void configure_default_solver()
            {
                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                KSPSetFromOptions(m_ksp);
            }

          protected:

            virtual void configure_solver()
            {
                configure_default_solver();
            }

          public:

            virtual void setup()
            {
                if (is_set_up())
                {
                    return;
                }

                if (assembly().undefined_unknown())
                {
                    std::cerr << "Undefined unknown(s) for this linear system. Please set the unknowns using the instruction '[solver].set_unknown(u);' or '[solver].set_unknowns(u1, u2...);'."
                              << std::endl;
                    assert(false && "Undefined unknown(s)");
                    exit(EXIT_FAILURE);
                }

                assembly().create_matrix(m_A);
                assembly().assemble_matrix(m_A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");

                // PetscBool is_symmetric;
                // MatIsSymmetric(m_A, 0, &is_symmetric);

                KSPSetOperators(m_ksp, m_A, m_A);
                PetscInt err = KSPSetUp(m_ksp);
                if (err != 0)
                {
                    std::cerr << "The setup of the solver failed!" << std::endl;
                    assert(false && "Failed solver setup");
                    exit(EXIT_FAILURE);
                }
                m_is_set_up = true;
            }

          protected:

            void prepare_rhs_and_solve(Vec& b, Vec& x)
            {
                // Update the right-hand side with the boundary conditions stored in the solution field
                assembly().enforce_bc(b);
                // Set to zero the right-hand side of the ghost equations
                assembly().enforce_projection_prediction(b);
                // Set to zero the right-hand side of the useless ghosts' equations
                assembly().add_0_for_useless_ghosts(b);
                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                // assert(check_nan_or_inf(b));

                solve_system(b, x);
            }

            void solve_system(Vec& b, Vec& x)
            {
                // Solve the system
                KSPSolve(m_ksp, b, x);

                KSPConvergedReason reason_code;
                KSPGetConvergedReason(m_ksp, &reason_code);
                if (reason_code < 0)
                {
                    using namespace std::string_literals;
                    const char* reason_text;
                    KSPGetConvergedReasonString(m_ksp, &reason_text);
                    std::cerr << "Divergence of the solver ("s + reason_text + ")" << std::endl;
                    // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                    // std::cout << std::endl;
                    // assert(check_nan_or_inf(b));
                    assert(false && "Divergence of the solver");
                    exit(EXIT_FAILURE);
                }
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
            }

          public:

            int iterations()
            {
                PetscInt n_iterations;
                KSPGetIterationNumber(m_ksp, &n_iterations);
                return n_iterations;
            }

            virtual void reset()
            {
                destroy_petsc_objects();
                m_is_set_up = false;
                configure_solver();
            }
        };

        template <class Assembly>
        class SingleFieldSolver : public SolverBase<Assembly>
        {
            using base_class = SolverBase<Assembly>;
            using scheme_t   = typename Assembly::scheme_t;
            using Field      = typename scheme_t::field_t;
            using Mesh       = typename Field::mesh_t;

            using base_class::assembly;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

          private:

            bool m_use_samurai_mg = false;
#ifdef ENABLE_MG
            GeometricMultigrid<Assembly> _samurai_mg;
#endif

          public:

            explicit SingleFieldSolver(const scheme_t& scheme)
                : base_class(scheme)
            {
                configure_solver();
            }

#ifdef ENABLE_MG
            void destroy_petsc_objects() override
            {
                base_class::destroy_petsc_objects();
                _samurai_mg.destroy_petsc_objects();
            }
#endif

          protected:

            void configure_solver() override
            {
                KSP user_ksp;
                KSPCreate(PETSC_COMM_SELF, &user_ksp);
                KSPSetFromOptions(user_ksp);
                PC user_pc;
                KSPGetPC(user_ksp, &user_pc);
                PCType user_pc_type;
                PCGetType(user_pc, &user_pc_type);
#ifdef ENABLE_MG
                m_use_samurai_mg = strcmp(user_pc_type, PCMG) == 0;
#endif
                KSPDestroy(&user_ksp);

                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                KSPSetFromOptions(m_ksp);
#ifdef ENABLE_MG
                if (m_use_samurai_mg)
                {
                    if constexpr (Mesh::dim > 2)
                    {
                        std::cerr << "Samurai Multigrid is not implemented for "
                                     "dim > 2."
                                  << std::endl;
                        assert(false);
                        exit(EXIT_FAILURE);
                    }
                    _samurai_mg = GeometricMultigrid(assembly(), assembly().mesh());
                    _samurai_mg.apply_as_pc(m_ksp);
                }
#endif
                m_is_set_up = false;
            }

          public:

            void set_unknown(Field& unknown)
            {
                assembly().set_unknown(unknown);
            }

            void setup() override
            {
                if (m_is_set_up)
                {
                    return;
                }
                if (assembly().undefined_unknown())
                {
                    std::cerr << "Undefined unknown for this linear system. Please set the unknown using the instruction '[solver].set_unknown(u);'."
                              << std::endl;
                    assert(false && "Undefined unknown");
                    exit(EXIT_FAILURE);
                }
                if (!m_use_samurai_mg)
                {
                    assembly().create_matrix(m_A);
                    assembly().assemble_matrix(m_A);
                    PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");

                    // PetscBool is_symmetric;
                    // MatIsSymmetric(m_A, 0, &is_symmetric);

                    KSPSetOperators(m_ksp, m_A, m_A);
                }
                KSPSetUp(m_ksp);
                m_is_set_up = true;
            }

            void solve(const Field& rhs)
            {
                if (!m_is_set_up)
                {
                    setup();
                }
                Vec b = create_petsc_vector_from(rhs);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b");
                Vec x = create_petsc_vector_from(assembly().unknown());
                this->prepare_rhs_and_solve(b, x);

                VecDestroy(&b);
                VecDestroy(&x);
            }

            void solve(Field& unknown, const Field& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

        /**
         * Block solver
         */
        template <bool monolithic, std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockSolver
        {
        };

        /**
         * Nested block solver
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockSolver<false, rows_, cols_, Operators...> : public SolverBase<NestedBlockAssembly<rows_, cols_, Operators...>>
        {
            using assembly_t = NestedBlockAssembly<rows_, cols_, Operators...>;
            using base_class = SolverBase<assembly_t>;
            using base_class::assembly;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

            using block_operator_t            = typename assembly_t::scheme_t;
            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr bool is_monolithic = false;

            explicit BlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
                configure_solver();
            }

          private:

            void configure_solver() override
            {
                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                // KSPSetFromOptions(m_ksp);
            }

          public:

            template <class... Fields>
            void set_unknowns(Fields&... unknowns)
            {
                auto unknown_tuple = assembly().block_operator().tie_unknowns(unknowns...);
                set_unknown(unknown_tuple);
            }

            template <class UnknownTuple>
            void set_unknown(UnknownTuple& unknown_tuple)
            {
                static_assert(std::tuple_size_v<UnknownTuple> == cols,
                              "The number of unknown fields passed to solve() must equal "
                              "the number of columns of the block operator.");
                assembly().set_unknown(unknown_tuple);
                assembly().reset();
            }

            void setup() override
            {
                if (m_is_set_up)
                {
                    return;
                }

                if (assembly().undefined_unknown())
                {
                    std::cerr << "Undefined unknowns for this linear system. Please set the unknowns using the instruction '[solver].set_unknowns(u1, u2...);'."
                              << std::endl;
                    assert(false && "Undefined unknowns");
                    exit(EXIT_FAILURE);
                }

                // assembly().reset();
                assembly().create_matrix(m_A);
                assembly().assemble_matrix(m_A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");
                // MatView(m_A, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                KSPSetOperators(m_ksp, m_A, m_A);

                // Set names to the petsc fields
                PC pc;
                KSPGetPC(m_ksp, &pc);
                IS is_fields[cols];
                MatNestGetISs(m_A, is_fields, NULL);
                auto field_names = assembly().field_names();
                for (std::size_t i = 0; i < cols; ++i)
                {
                    PCFieldSplitSetIS(pc, field_names[i].c_str(), is_fields[i]);
                }

                KSPSetFromOptions(m_ksp);
                PCSetUp(pc);
                // KSPSetUp(m_ksp); // Here, PETSc fails for some reason.

                m_is_set_up = true;
            }

            template <class... Fields>
            void solve(Fields&... rhs_fields)
            {
                auto rhs_tuple = assembly().block_operator().tie_rhs(rhs_fields...);
                // static_assert(std::tuple_size_v<RHSTuple> == rows,
                //                   "The number of source fields passed to solve() must equal "
                //                   "the number of rows of the block operator.");

                if (!m_is_set_up)
                {
                    setup();
                }
                Vec b = assembly().create_rhs_vector(rhs_tuple);
                Vec x = assembly().create_solution_vector();
                this->prepare_rhs_and_solve(b, x);

                VecDestroy(&b);
                VecDestroy(&x);
            }
        };

        /**
         * Monolithic block solver
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockSolver<true, rows_, cols_, Operators...> : public SolverBase<MonolithicBlockAssembly<rows_, cols_, Operators...>>
        {
            using assembly_t = MonolithicBlockAssembly<rows_, cols_, Operators...>;
            using base_class = SolverBase<assembly_t>;
            using base_class::assembly;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

            using block_operator_t = typename assembly_t::scheme_t;

            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr bool is_monolithic = true;

            explicit BlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
            }

            template <class... Fields>
            void set_unknowns(Fields&... unknowns)
            {
                auto unknown_tuple = assembly().block_operator().tie_unknowns(unknowns...);
                set_unknown(unknown_tuple);
            }

            template <class UnknownTuple>
            void set_unknown(UnknownTuple& unknown_tuple)
            {
                static_assert(std::tuple_size_v<UnknownTuple> == cols,
                              "The number of unknown fields passed to solve() must equal "
                              "the number of columns of the block operator.");
                assembly().set_unknown(unknown_tuple);
                assembly().reset();
            }

            template <class... Fields>
            void solve(Fields&... rhs_fields)
            {
                auto rhs_tuple = assembly().block_operator().tie_rhs(rhs_fields...);
                // static_assert(std::tuple_size_v<RHSTuple> == rows,
                //                   "The number of source fields passed to solve() must equal "
                //                   "the number of rows of the block operator.");

                if (!m_is_set_up)
                {
                    this->setup();
                }

                Vec b = assembly().create_rhs_vector(rhs_tuple);
                Vec x = assembly().create_solution_vector();
                this->prepare_rhs_and_solve(b, x);

                assembly().update_unknowns(x);

                VecDestroy(&b);
                VecDestroy(&x);
            }
        };

        /**
         * Helper functions
         */

        template <class Scheme>
        auto make_solver(const Scheme& scheme)
        {
            return SingleFieldSolver<Assembly<Scheme>>(scheme);
        }

        template <class Scheme>
        void solve(const Scheme& scheme, typename Scheme::field_t& unknown, const typename Scheme::field_t& rhs)
        {
            auto solver = make_solver(scheme);
            solver.solve(unknown, rhs);
        }

        template <bool monolithic, std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return BlockSolver<monolithic, rows, cols, Operators...>(block_operator);
        }

        template <std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            static constexpr bool default_monolithic = true;
            return make_solver<default_monolithic, rows, cols, Operators...>(block_operator);
        }

    } // end namespace petsc
} // end namespace samurai
