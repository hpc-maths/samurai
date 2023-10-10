#pragma once

#include "fv/cell_based_scheme_assembly.hpp"
#include "fv/flux_based_scheme_assembly.hpp"
#include "fv/scheme_operators_assembly.hpp"
#include "utils.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        template <class Assembly>
        class NonLinearSolverBase
        {
            using scheme_t = typename Assembly::scheme_t;
            using field_t  = typename scheme_t::field_t;

          protected:

            Assembly m_assembly;
            SNES m_snes      = nullptr;
            Mat m_J          = nullptr;
            bool m_is_set_up = false;

          public:

            // Vec m_b;        // REMOVE
            // field_t* m_rhs; // REMOVE

            explicit NonLinearSolverBase(const scheme_t& scheme)
                : m_assembly(scheme)
            {
                configure_default_solver();
            }

            virtual ~NonLinearSolverBase()
            {
                destroy_petsc_objects();
            }

            virtual void destroy_petsc_objects()
            {
                if (m_J)
                {
                    MatDestroy(&m_J);
                    m_J = nullptr;
                }
                if (m_snes)
                {
                    SNESDestroy(&m_snes);
                    m_snes = nullptr;
                }
            }

            NonLinearSolverBase& operator=(const NonLinearSolverBase& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_snes      = other.m_snes;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                }
                return *this;
            }

            NonLinearSolverBase& operator=(NonLinearSolverBase&& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_assembly  = other.m_assembly;
                    this->m_snes      = other.m_snes;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                    other.m_snes      = nullptr; // Prevent SNES destruction when 'other' object is destroyed
                    other.m_J         = nullptr;
                    other.m_is_set_up = false;
                }
                return *this;
            }

            SNES& Snes()
            {
                return m_snes;
            }

            // KSP& Ksp()
            // {
            //     return m_ksp;
            // }

            bool is_set_up()
            {
                return m_is_set_up;
            }

            auto& assembly()
            {
                return m_assembly;
            }

            auto& scheme()
            {
                return assembly().scheme();
            }

          private:

            void configure_default_solver()
            {
                SNESCreate(PETSC_COMM_SELF, &m_snes);
                SNESSetType(m_snes, SNESNEWTONLS);
                SNESSetFromOptions(m_snes);
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
                    std::cerr << "Undefined unknown(s) for this non-linear system. Please set the unknowns using the instruction '[solver].set_unknown(u);' or '[solver].set_unknowns(u1, u2...);'."
                              << std::endl;
                    assert(false && "Undefined unknown(s)");
                    exit(EXIT_FAILURE);
                }

                // Non-linear function
                SNESSetFunction(m_snes, nullptr, PETSC_nonlinear_function, this);

                // Jacobian matrix
                assembly().create_matrix(m_J);
                assembly().assemble_matrix(m_J);
                SNESSetJacobian(m_snes, m_J, m_J, PETSC_jacobian_function, this);

                m_is_set_up = true;
            }

            static inline void set_0_in_each_ghost(field_t& field)
            {
                using mesh_id_t = typename field_t::mesh_t::mesh_id_t;

                auto& mesh = field.mesh();
                for (std::size_t level = mesh.min_level(); level <= mesh.min_level(); ++level)
                {
                    auto ghosts = difference(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::cells][level]);
                    for_each_interval(ghosts,
                                      [&](auto l, const auto& i, const auto& index)
                                      {
                                          field(l, i, index) = 0;
                                      });
                }
            }

          private:

            static PetscErrorCode PETSC_nonlinear_function(SNES /*snes*/, Vec x, Vec f, void* ctx)
            {
                // const PetscScalar* xx;
                // VecGetArrayRead(x, &xx);
                // field_t x_field("Newton iterate", assembly().unknown().mesh(), const_cast<PetscScalar*>(xx));

                auto self = reinterpret_cast<NonLinearSolverBase*>(ctx); // this

                auto& mesh = self->assembly().unknown().mesh();
                auto iter  = self->iterations();

                // assembly().enforce_projection_prediction(x);

                field_t x_field("newton", self->assembly().unknown().mesh());
                copy(x, x_field); // This is really bad... A field structure wrapped around the data of the Petsc vector is what we want
                // Transfer B.C.
                std::transform(self->assembly().unknown().get_bc().cbegin(),
                               self->assembly().unknown().get_bc().cend(),
                               std::back_inserter(x_field.get_bc()),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });

                // std::cout << x_field << std::endl;

                // PetscScalar* ff;
                // VecGetArray(f, &ff);
                // field_t f_field("f", assembly().unknown().mesh(), ff);

                // f_field = f(x_field)

                samurai::save(std::filesystem::current_path(), fmt::format("newton_x_ite_{}", iter), mesh, x_field);

                // double x_norm;
                // VecNorm(x, NORM_2, &x_norm);
                // std::cout << "x_norm = " << x_norm << std::endl;

                // update_ghost_mr(x_field);
                auto f_field = self->scheme()(x_field); // apply explicit scheme

                // std::cout << f_field << std::endl;

                samurai::save(std::filesystem::current_path(), fmt::format("newton_ite_{}", iter), f_field.mesh(), x_field, f_field);

                // double f_norm;
                // VecNorm(f, NORM_2, &f_norm);
                // std::cout << "f_norm (before) = " << f_norm << std::endl;

                set_0_in_each_ghost(f_field);
                copy(f_field, f);

                // VecView(f, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                // Update the right-hand side with the boundary conditions stored in the solution field
                self->assembly().enforce_bc(f);
                // Set to zero the right-hand side of the ghost equations
                self->assembly().enforce_projection_prediction(f);
                // Set to zero the right-hand side of the useless ghosts' equations
                self->assembly().add_0_for_useless_ghosts(f);

                // VecZeroEntries(f);

                /**
                 *  Reproduction of the Newton algorithm:
                 */
                // Vec b;
                // VecDuplicate(f, &b);
                // copy(f_field, f);
                // VecAXPY(f, -1.0, self->m_b); // f = f-b

                // Mat J; // Jacobian matrix
                // self->assembly().create_matrix(J);
                // PETSC_jacobian_function(nullptr, x, J, J, self);

                // KSP ksp;
                // KSPCreate(PETSC_COMM_SELF, &ksp);
                // KSPSetFromOptions(ksp);
                // PetscCall(KSPSetOperators(ksp, J, J));
                // Vec y;
                // VecDuplicate(f, &y);
                // PetscCall(KSPSolve(ksp, f, y)); // y = J^-1 f
                // field_t y_field("y", mesh);
                // copy(y, y_field);

                // Vec w;
                // VecDuplicate(y, &w);
                // VecWAXPY(w, -1, y, x); // w = x - y
                // field_t w_field("w", mesh);
                // copy(w, w_field);

                // samurai::save(std::filesystem::current_path(), fmt::format("newton2_ite_{}", iter), mesh, x_field, y_field, w_field);

                // auto f2_field = self->scheme()(w_field); // apply explicit scheme
                // Vec f2        = create_petsc_vector_from(f2_field);

                // samurai::save(std::filesystem::current_path(), fmt::format("newton3_ite_{}", iter), mesh, w_field, f2_field,
                // *self->m_rhs);

                // set_0_in_each_ghost(f2_field);
                // // Update the right-hand side with the boundary conditions stored in the solution field
                // self->assembly().enforce_bc(f2);
                // // Set to zero the right-hand side of the ghost equations
                // self->assembly().enforce_projection_prediction(f2);
                // // Set to zero the right-hand side of the useless ghosts' equations
                // self->assembly().add_0_for_useless_ghosts(f2);

                // samurai::save(std::filesystem::current_path(), fmt::format("newton4_ite_{}", iter), mesh, w_field, f2_field,
                // *self->m_rhs);

                // VecAXPY(f2, -1.0, self->m_b); // f2 = f2-b

                // samurai::save(std::filesystem::current_path(), fmt::format("newton5_ite_{}", iter), mesh, w_field, f2_field,
                // *self->m_rhs);

                // double f2_norm;
                // VecNorm(f2, NORM_2, &f2_norm);
                // std::cout << "f2_norm " << f2_norm << std::endl;

                /**
                 * --------------------------------------------------------------------------------------
                 */

                // VecNorm(f, NORM_2, &f_norm);
                // std::cout << "f_norm (after) = " << f_norm << std::endl;

                // std::cout << f_field << std::endl;
                // VecView(f2, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                // VecRestoreArrayRead(x, &xx);
                // VecRestoreArray(f, &ff);

                return 0; // PETSC_SUCCESS
            }

            static PetscErrorCode PETSC_jacobian_function(SNES /*snes*/, Vec x, Mat jac, Mat B, void* ctx)
            {
                // Here, jac = B = this.m_J
                // Petsc recommends we assemble B.

                auto self = reinterpret_cast<NonLinearSolverBase*>(ctx); // this

                // const PetscScalar* xx;
                // VecGetArrayRead(x, &xx);
                // field_t x_field("x", assembly().unknown().mesh(), const_cast<PetscScalar*>(xx));
                field_t x_field("newton_jac_x", self->assembly().unknown().mesh());
                copy(x, x_field); // This is really bad... A field structure wrapped around the data of the Petsc vector is what we want

                // Transfer B.C. so that the assembly process has B.C. to enforce in the matrix
                std::transform(self->assembly().unknown().get_bc().cbegin(),
                               self->assembly().unknown().get_bc().cend(),
                               std::back_inserter(x_field.get_bc()),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });
                update_bc(x_field); // Not sure if necessary

                // Save unknown...
                auto real_system_unknown = self->assembly().unknown_ptr();
                // and replace it with the current Newton iterate (so that the Jacobian matrix is computed at that specific point)
                self->assembly().set_unknown(x_field);

                MatZeroEntries(B);
                self->assembly().assemble_matrix(B);
                PetscObjectSetName(reinterpret_cast<PetscObject>(B), "Jacobian");
                if (jac != B)
                {
                    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
                }

                // MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                // Put back the real unknown: we need its B.C. for the evaluation of the non-linear function
                self->assembly().set_unknown(*real_system_unknown);

                // VecRestoreArrayRead(x, &xx);

                return 0; // PETSC_SUCCESS
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
                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                // assert(check_nan_or_inf(b));

                solve_system(b, x);
            }

            void solve_system(Vec& b, Vec& x)
            {
                // this->m_b = b; // TO REMOVE

                // Solve the system
                SNESSolve(m_snes, b, x);
                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                SNESConvergedReason reason_code;
                SNESGetConvergedReason(m_snes, &reason_code);
                if (reason_code < 0)
                {
                    using namespace std::string_literals;
                    const char* reason_text;
                    SNESGetConvergedReasonString(m_snes, &reason_text);
                    std::cerr << "Divergence of the non-linear solver ("s + reason_text + ")" << std::endl;
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
                SNESGetIterationNumber(m_snes, &n_iterations);
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
        class SingleFieldNonLinearSolver : public NonLinearSolverBase<Assembly>
        {
            using base_class = NonLinearSolverBase<Assembly>;
            using scheme_t   = typename Assembly::scheme_t;
            using Field      = typename scheme_t::field_t;
            using Mesh       = typename Field::mesh_t;

            using base_class::assembly;
            using base_class::m_is_set_up;
            using base_class::m_J;
            // using base_class::m_ksp;

          public:

            explicit SingleFieldNonLinearSolver(const scheme_t& scheme)
                : base_class(scheme)
            {
                // configure_solver();
            }

          protected:

            // void configure_solver() override
            // {
            //     KSP user_ksp;
            //     KSPCreate(PETSC_COMM_SELF, &user_ksp);
            //     KSPSetFromOptions(user_ksp);
            //     PC user_pc;
            //     KSPGetPC(user_ksp, &user_pc);
            //     PCType user_pc_type;
            //     PCGetType(user_pc, &user_pc_type);

            //     KSPDestroy(&user_ksp);

            //     KSPCreate(PETSC_COMM_SELF, &m_ksp);
            //     KSPSetFromOptions(m_ksp);
            //     m_is_set_up = false;
            // }

          public:

            void set_unknown(Field& unknown)
            {
                assembly().set_unknown(unknown);
            }

            // void setup() override
            // {
            //     if (m_is_set_up)
            //     {
            //         return;
            //     }
            //     if (assembly().undefined_unknown())
            //     {
            //         std::cerr << "Undefined unknown for this linear system. Please set the unknown using the instruction
            //         '[solver].set_unknown(u);'."
            //                   << std::endl;
            //         assert(false && "Undefined unknown");
            //         exit(EXIT_FAILURE);
            //     }

            //     assembly().create_matrix(m_A);
            //     assembly().assemble_matrix(m_A);
            //     PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");

            //     KSPSetOperators(m_ksp, m_A, m_A);

            //     KSPSetUp(m_ksp);
            //     m_is_set_up = true;
            // }

            void solve(/*const*/ Field& rhs)
            {
                base_class::set_0_in_each_ghost(rhs);
                // this->m_rhs = &rhs;

                if (!m_is_set_up)
                {
                    this->setup();
                }
                Vec b = create_petsc_vector_from(rhs);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b");
                Vec x = create_petsc_vector_from(assembly().unknown());
                this->prepare_rhs_and_solve(b, x);

                VecDestroy(&b);
                VecDestroy(&x);
            }

            void solve(Field& unknown, /*const*/ Field& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

        /**
         * Helper functions
         */

        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type == SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return SingleFieldNonLinearSolver<Assembly<Scheme>>(scheme);
        }

    } // end namespace petsc
} // end namespace samurai
