#pragma once
#include "fv/cell_based_scheme_assembly.hpp"
#include "fv/flux_based_scheme_assembly.hpp"
#include "fv/operator_sum_assembly.hpp"
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
                // assembly().assemble_matrix(m_J);
                SNESSetJacobian(m_snes, m_J, m_J, PETSC_jacobian_function, this);

                m_is_set_up = true;
            }

          private:

            static PetscErrorCode PETSC_nonlinear_function(SNES /*snes*/, Vec x, Vec f, void* ctx)
            {
                // const char* x_name;
                // PetscObjectGetName(reinterpret_cast<PetscObject>(x), &x_name);
                // const char* f_name;
                // PetscObjectGetName(reinterpret_cast<PetscObject>(f), &f_name);

                auto self      = reinterpret_cast<NonLinearSolverBase*>(ctx); // this
                auto& assembly = self->assembly();
                auto& mesh     = assembly.unknown().mesh();

                // Wrap a field structure around the data of the Petsc vector x
                field_t x_field("newton", mesh);
                copy(x, x_field); // This is really bad... TODO: create a field constructor that takes a double*

                // Transfer B.C. to the new field (required to be able to apply the explicit scheme)
                std::transform(assembly.unknown().get_bc().cbegin(),
                               assembly.unknown().get_bc().cend(),
                               std::back_inserter(x_field.get_bc()),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });

                // Apply explicit scheme
                update_ghost_mr(x_field);
                auto f_field = self->scheme()(x_field);

                copy(f_field, f);
                self->prepare_rhs(f);
                return 0; // PETSC_SUCCESS
            }

            static PetscErrorCode PETSC_jacobian_function(SNES /*snes*/, Vec x, Mat jac, Mat B, void* ctx)
            {
                // Here, jac = B = this.m_J

                auto self      = reinterpret_cast<NonLinearSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                // Wrap a field structure around the data of the Petsc vector x
                field_t x_field("newton_jac_x", assembly.unknown().mesh());
                copy(x, x_field); // This is really bad... TODO: create a field constructor that takes a double*

                // Transfer B.C. to the new field,
                // so that the assembly process has B.C. to enforce in the matrix
                std::transform(assembly.unknown().get_bc().cbegin(),
                               assembly.unknown().get_bc().cend(),
                               std::back_inserter(x_field.get_bc()),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });
                update_bc(x_field); // Not sure if necessary

                // Save unknown...
                auto real_system_unknown = assembly.unknown_ptr();
                // and replace it with the current Newton iterate (so that the Jacobian matrix is computed at that specific point)
                assembly.set_unknown(x_field);

                // Assembly of the Jacobian matrix.
                // In this case, jac = B, but Petsc recommends we assemble B for more general cases.
                MatZeroEntries(B);
                assembly.assemble_matrix(B);
                PetscObjectSetName(reinterpret_cast<PetscObject>(B), "Jacobian");
                if (jac != B)
                {
                    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
                }

                // MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                // Put back the real unknown: we need its B.C. for the evaluation of the non-linear function
                assembly.set_unknown(*real_system_unknown);

                return 0; // PETSC_SUCCESS
            }

          protected:

            void prepare_rhs(Vec& b)
            {
                assembly().set_0_for_all_ghosts(b);
                // Update the right-hand side with the boundary conditions stored in the solution field
                assembly().enforce_bc(b);
                // Set to zero the right-hand side of the ghost equations
                // assembly().enforce_projection_prediction(b);
                // Set to zero the right-hand side of the useless ghosts' equations
                // assembly().set_0_for_useless_ghosts(b);

                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                // assert(check_nan_or_inf(b));
            }

            void prepare_rhs_and_solve(Vec& b, Vec& x)
            {
                prepare_rhs(b);
                solve_system(b, x);
            }

            void solve_system(Vec& b, Vec& x)
            {
                // Solve the system
                SNESSolve(m_snes, b, x);

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

        template <class Scheme>
        class NonLinearSolver : public NonLinearSolverBase<Assembly<Scheme>>
        {
            using base_class = NonLinearSolverBase<Assembly<Scheme>>;

          public:

            using scheme_t = Scheme;
            using Field    = typename scheme_t::field_t;
            using Mesh     = typename Field::mesh_t;

            using base_class::assembly;
            using base_class::m_is_set_up;
            using base_class::m_J;

            explicit NonLinearSolver(const scheme_t& scheme)
                : base_class(scheme)
            {
            }

            void set_unknown(Field& unknown)
            {
                assembly().set_unknown(unknown);
            }

            void solve(Field& rhs)
            {
                if (!m_is_set_up)
                {
                    this->setup();
                }
                Vec b = create_petsc_vector_from(rhs);
                Vec x = create_petsc_vector_from(assembly().unknown());
                this->prepare_rhs_and_solve(b, x);

                VecDestroy(&b);
                VecDestroy(&x);
            }

            void solve(Field& unknown, Field& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

    } // end namespace petsc
} // end namespace samurai
