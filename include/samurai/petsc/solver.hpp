#pragma once

// #define ENABLE_MG

#include "block_assembly.hpp"
#ifdef ENABLE_MG
#include "multigrid/petsc/GeometricMultigrid.hpp"
#else
#include "utils.hpp"
#endif

namespace samurai
{
    namespace petsc
    {
        template <class Dsctzr>
        class Solver
        {
            using Mesh  = typename Dsctzr::Mesh;
            using Field = typename Dsctzr::field_t;

          private:

            Dsctzr& m_discretizer;
            KSP m_ksp             = nullptr;
            bool m_use_samurai_mg = false;
            Mat m_A               = nullptr;
            bool m_is_set_up      = false;
#ifdef ENABLE_MG
            GeometricMultigrid<Dsctzr> _samurai_mg;
#endif

          public:

            Solver(Dsctzr& discretizer)
                : m_discretizer(discretizer)
            {
                create_solver();
            }

            ~Solver()
            {
                destroy_petsc_objects();
            }

            void destroy_petsc_objects()
            {
#ifdef ENABLE_MG
                _samurai_mg.destroy_petsc_objects();
#endif
                if (m_A)
                {
                    MatDestroy(&m_A);
                    m_A = nullptr;
                }
                /*if (m_ksp)
                {
                    KSPDestroy(&m_ksp);
                    m_ksp = nullptr;
                }*/
            }

            KSP& Ksp()
            {
                return m_ksp;
            }

        private:
            void create_solver()
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
                    _samurai_mg = GeometricMultigrid(m_discretizer, m_discretizer.mesh());
                    _samurai_mg.apply_as_pc(m_ksp);
                }
    #endif
                m_is_set_up = false;
            }

          public:

            void setup()
            {
                if (m_is_set_up)
                {
                    return;
                }
                if (!m_use_samurai_mg)
                {
                    m_discretizer.create_matrix(m_A);
                    m_discretizer.assemble_matrix(m_A);
                    PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");
                    KSPSetOperators(m_ksp, m_A, m_A);
                }
                KSPSetUp(m_ksp);
                m_is_set_up = true;
            }

            void solve(const Field& source)
            {
                if (!m_is_set_up)
                {
                    setup();
                }

                // Create right-hand side vector from the source field
                Vec b = create_petsc_vector_from(source);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b),
                                   "b"); // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                                         // std::cout << std::endl;

                // Update the right-hand side with the boundary conditions
                // stored in the solution field
                m_discretizer.enforce_bc(b);                    // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                                                                // std::cout << std::endl;
                m_discretizer.enforce_projection_prediction(b); // VecView(b,
                                                                // PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                                                                // std::cout << std::endl;

                // Create the solution vector
                Vec x = create_petsc_vector_from(m_discretizer.unknown());

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
                    assert(false);
                    exit(EXIT_FAILURE);
                }
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl;

                VecDestroy(&b);
                VecDestroy(&x);
            }

            int iterations()
            {
                PetscInt n_iterations;
                KSPGetIterationNumber(m_ksp, &n_iterations);
                return n_iterations;
            }

            void reset()
            {
                destroy_petsc_objects();
                create_solver();
            }
        };

        template <class Dsctzr>
        Solver<Dsctzr> make_solver(Dsctzr& discretizer)
        {
            return Solver<Dsctzr>(discretizer);
        }

        template <class Dsctzr>
        void solve(Dsctzr& discretizer, const typename Dsctzr::field_t& rhs)
        {
            Solver<Dsctzr> solver(discretizer);
            solver.solve(rhs);
        }

        /**
         * PETSc block solver
        */
        template <class Dsctzr>
        class BlockSolver
        {
            static constexpr int rows = Dsctzr::n_rows;
            static constexpr int cols = Dsctzr::n_cols;
        private:
            Dsctzr& m_discretizer;
            KSP m_ksp        = nullptr;
            Mat m_A          = nullptr;
            bool m_is_set_up = false;
        public:
            BlockSolver(Dsctzr& discretizer)
            : m_discretizer(discretizer)
            {
                create_solver();
            }

            ~BlockSolver()
            {
                destroy_petsc_objects();
            }

            void destroy_petsc_objects()
            {
                if (m_A)
                {
                    MatDestroy(&m_A);
                    m_A = nullptr;
                }
                /*if (m_ksp)
                {
                    KSPDestroy(&m_ksp);
                    m_ksp = nullptr;
                }*/
            }

            KSP& Ksp()
            {
                return m_ksp;
            }

          private:

            void create_solver()
            {
                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                // KSPSetFromOptions(m_ksp);
            }

          public:

            void setup()
            {
                if (m_is_set_up)
                {
                    return;
                }

                m_discretizer.create_matrix(m_A);
                m_discretizer.assemble_matrix(m_A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(m_A),
                                   "A"); // MatView(m_A,
                                         // PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                                         // std::cout << std::endl;
                KSPSetOperators(m_ksp, m_A, m_A);

                // Set names to the petsc fields
                PC pc;
                KSPGetPC(m_ksp, &pc);
                IS is_fields[cols];
                MatNestGetISs(m_A, is_fields, NULL);
                auto field_names = m_discretizer.field_names();
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
            void solve(const Fields&... sources)
            {
                auto s = std::tuple<const Fields&...>(sources...);
                solve(s);
            }

            template <class... Fields>
            void solve(const std::tuple<Fields&...>& sources)
            {
                static_assert(sizeof...(Fields) == rows,
                              "The number of source fields passed to solve() must equal "
                              "the number of rows of the block operator.");

                if (!m_is_set_up)
                {
                    setup();
                }

                // Create a right-hand side block-vector from the source fields
                std::array<Vec, rows> b_blocks;
                std::size_t i = 0;
                for_each(sources,
                         [&](auto& s)
                         {
                             b_blocks[i] = create_petsc_vector_from(s);
                             PetscObjectSetName(reinterpret_cast<PetscObject>(b_blocks[i]), s.name().c_str());
                             i++;
                         });
                Vec b;
                VecCreateNest(PETSC_COMM_SELF, rows, NULL, b_blocks.data(), &b);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side"); //VecView(b_blocks[0], PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                //assert(check_nan_or_inf(b));

                // Update the right-hand side with the boundary conditions stored in the solution field
                m_discretizer.enforce_projection_prediction(b_blocks);                   //VecView(b_blocks[0], PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                m_discretizer.enforce_bc(b_blocks);                                      //VecView(b_blocks[0], PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                //m_discretizer.add_0_for_useless_ghosts(b_blocks);                        //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                //assert(check_nan_or_inf(b));

                // Create the solution vector
                std::array<Vec, cols> x_blocks = m_discretizer.create_solution_vectors();
                Vec x;
                VecCreateNest(PETSC_COMM_SELF, cols, NULL, x_blocks.data(), &x);
                PetscObjectSetName(reinterpret_cast<PetscObject>(x),
                                   "solution"); // VecView(x,
                                                // PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                                                // std::cout << std::endl;

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
                    //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                    assert(false && "Divergence of the solver");
                    exit(EXIT_FAILURE);
                }
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl;

                VecDestroy(&b);
                VecDestroy(&x);
            }

            int iterations()
            {
                PetscInt n_iterations;
                KSPGetIterationNumber(m_ksp, &n_iterations);
                return n_iterations;
            }

            void reset()
            {
                destroy_petsc_objects();
                create_solver();
            }
        };

        template <class Dsctzr>
        auto make_block_solver(Dsctzr& discretizer)
        {
            return BlockSolver<Dsctzr>(discretizer);
        }

    } // end namespace petsc
} // end namespace samurai