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

            static constexpr bool is_block_solver = IsBlockOperator<scheme_t>;

            using output_field_t = typename scheme_t::output_field_t;

            // Helper to conditionally access output_field_no_ref_t only when it exists
            template <class Scheme, bool is_block>
            struct worker_output_field_type_helper
            {
                using type = output_field_t;
            };

            template <class Scheme>
            struct worker_output_field_type_helper<Scheme, true>
            {
                using type = typename Scheme::output_field_no_ref_t;
            };

            // If not a block solver, worker_output_field_t = output_field_t
            using worker_output_field_t = typename worker_output_field_type_helper<scheme_t, is_block_solver>::type;

          protected:

            Assembly m_assembly;
            SNES m_snes                   = nullptr;
            Mat m_J                       = nullptr;
            bool m_is_set_up              = false;
            bool m_reuse_allocated_matrix = false;

            worker_output_field_t m_worker_output_field;

          public:

            // User callback to configure the solver
            std::function<void(SNES&, KSP&, PC&)> configure                   = nullptr;
            std::function<void(SNES&, KSP&, PC&, Mat&)> after_matrix_assembly = nullptr;
            std::function<void(const worker_output_field_t&)> monitor         = nullptr;

            explicit NonLinearSolverBase(const scheme_t& scheme)
                : m_assembly(scheme)
            {
                SNESCreate(PETSC_COMM_WORLD, &m_snes);
            }

            virtual ~NonLinearSolverBase()
            {
                _destroy_petsc_objects();
            }

          private:

            void _destroy_petsc_objects()
            {
                if (m_J)
                {
                    m_assembly.destroy_local_to_global_mappings(m_J);
                    MatDestroy(&m_J);
                    m_J                      = nullptr;
                    m_reuse_allocated_matrix = false;
                }
                if (m_snes)
                {
                    SNESDestroy(&m_snes);
                    m_snes = nullptr;
                }
            }

          public:

            virtual void destroy_petsc_objects()
            {
                _destroy_petsc_objects();
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

          protected:

            virtual void default_solver_configuration()
            {
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
                if (!m_reuse_allocated_matrix)
                {
                    assembly().create_matrix(m_J);
                }

                // Jacobian function
                SNESSetJacobian(m_snes, m_J, m_J, PETSC_jacobian_function, this);

                // Sends the successive residuals to the user if he monitors the convergence
                if (monitor)
                {
                    SNESMonitorSet(m_snes, PETSC_monitor, this, nullptr);
                }

                KSP ksp;
                PC pc;
                SNESGetKSP(m_snes, &ksp);
                KSPGetPC(ksp, &pc);
                if (configure)
                {
                    configure(m_snes, ksp, pc);
                }
                SNESSetFromOptions(m_snes);

                m_is_set_up = true;
            }

          private:

            static PetscErrorCode PETSC_nonlinear_function(SNES /*snes*/, Vec x, Vec f, void* ctx)
            {
                times::timers.stop("nonlinear system solve");

                auto self      = reinterpret_cast<NonLinearSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                if constexpr (!is_block_solver)
                {
                    // Ideally, we would like to wrap a field structure around the data of the Petsc vectors x and f,
                    // but we don't have such a Field constructor.
                    // So, instead, we use worker fields and copy the data.
                    auto& x_field = assembly.unknown();          // for x, we reuse the unknown field
                    auto& f_field = self->m_worker_output_field; // for f, we use an actual worker field

                    assembly.copy_unknown(x, x_field);
                    // x_field.ghosts_updated() = false;

                    // Apply explicit scheme: f = scheme(x)
                    f_field.fill(0); // initialize to zero because we accumulate the results
                    self->scheme().apply(f_field, x_field);

                    // Vec updated_x = assembly.create_solution_vector(x_field); // update_x is x with ghosts updated
                    //  assembly.copy_unknown(x_field, updated_x);
                    // update_ghost_mr(f_field);

                    // Copy the result into the Petsc vector f
                    assembly.copy_rhs(f_field, f);
                    // Set to zero the right-hand side of the ghost equations and apply BCs attached to the unknown field
                    self->prepare_rhs(x, f);
                }
                else
                {
                    assembly.update_unknowns(x);                  // for x, we reuse the unknown fields
                    auto& f_fields = self->m_worker_output_field; // for f, we use an actual worker field, which is a tuple of fields

                    // Apply explicit scheme: f = scheme(x)
                    for_each(f_fields,
                             [](auto& f_field)
                             {
                                 f_field.fill(0); // initialize to zero because we accumulate the results
                             });

                    auto unknown_tuple = assembly.unknown(); // tuple containing references to the unknown fields
                    assembly.block_operator().apply(f_fields, unknown_tuple);

                    // Copy the result into the Petsc vector f
                    assembly.copy_rhs(f_fields, f);
                    // Set to zero the right-hand side of the ghost equations and apply BCs attached to the unknown field
                    self->prepare_rhs(x, f);
                }

#ifdef SAMURAI_CHECK_NAN
                assert(check_nan_or_inf(f));
#endif
                times::timers.start("nonlinear system solve");
                return PETSC_SUCCESS;
            }

            static PetscErrorCode PETSC_jacobian_function(SNES snes, Vec x, Mat jac, Mat B, void* ctx)
            {
                times::timers.stop("nonlinear system solve");

                // Here, jac = B = this.m_J

                auto self      = reinterpret_cast<NonLinearSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                // Ideally, we would like to wrap a field structure around the data of the Petsc vector x,
                // but we don't have such a Field constructor.
                // So, instead, we reuse the unknown field and copy the data.
                if constexpr (!is_block_solver)
                {
                    assembly.copy_unknown(x, assembly.unknown());
                }
                else
                {
                    assembly.update_unknowns(x);
                }

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

                if (self->after_matrix_assembly)
                {
                    KSP ksp;
                    PC pc;
                    SNESGetKSP(snes, &ksp);
                    KSPGetPC(ksp, &pc);
                    self->after_matrix_assembly(snes, ksp, pc, B);
                }

                // MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                // std::cout << std::endl;

                times::timers.start("nonlinear system solve");
                return PETSC_SUCCESS;
            }

            static PetscErrorCode PETSC_monitor(SNES snes, PetscInt it, PetscReal /* rnorm */, void* ctx)
            {
                Vec r;
                SNESGetFunction(snes, &r, NULL, NULL);

                auto self      = reinterpret_cast<NonLinearSolverBase*>(ctx); // this
                auto& assembly = self->assembly();

                auto& r_field = self->m_worker_output_field;
                assembly.copy_rhs(r, r_field);
                if (self->monitor)
                {
                    self->monitor(r_field);
                }

                if constexpr (!is_block_solver)
                {
                    samurai::save(fs::current_path(), fmt::format("snes_residual_{}", it), {true, true}, r_field.mesh(), r_field);
                }
                else
                {
                    static constexpr std::size_t cols = Assembly::cols;
                    static_for<0, cols>::apply(
                        [&](auto col)
                        {
                            auto& r_field = std::get<col>(self->m_worker_output_field);
                            samurai::save(fs::current_path(),
                                          fmt::format("snes_residual_{}_{}", r_field.name(), it),
                                          {true, true},
                                          r_field.mesh(),
                                          r_field);
                        });
                }

                return PETSC_SUCCESS;
            }

          protected:

            void prepare_rhs(Vec& /* x */, Vec& b)
            {
                // assembly().copy_values_for_all_ghosts(x, b);
                assembly().set_0_for_all_ghosts(b);
                // Update the right-hand side with the boundary conditions stored in the solution field
                assembly().enforce_bc(b);
                // Set to zero the right-hand side of the ghost equations
                // assembly().enforce_projection_prediction(b);
                // Set to zero the right-hand side of the useless ghosts' equations
                // assembly().set_0_for_useless_ghosts(b);

                // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                // std::cout << std::endl;
                // assert(check_nan_or_inf(b));
            }

            void solve_system(Vec& x, const Vec& b)
            {
#ifdef SAMURAI_CHECK_NAN
                assert(check_nan_or_inf(x));
                assert(check_nan_or_inf(b));
#endif
                // Solve the system
                times::timers.start("nonlinear system solve");
                SNESSolve(m_snes, b, x);
                times::timers.stop("nonlinear system solve");

                SNESConvergedReason reason_code;
                SNESGetConvergedReason(m_snes, &reason_code);
                if (reason_code < 0)
                {
                    using namespace std::string_literals;
                    const char* reason_text;
                    SNESGetConvergedReasonString(m_snes, &reason_text);
                    std::cerr << "Divergence of the non-linear solver ("s + reason_text + ")" << std::endl;
                    // VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD));
                    // std::cout << std::endl;
                    // assert(check_nan_or_inf(b));
                    assert(false && "Divergence of the solver");
                    exit(EXIT_FAILURE);
                }
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)); std::cout << std::endl;
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
                SNESCreate(PETSC_COMM_WORLD, &m_snes);
                m_assembly.is_set_up(false);
                m_is_set_up = false;
                default_solver_configuration();
            }

            void set_scheme(const scheme_t& s)
            {
                m_assembly.set_scheme(s);
                if (m_snes)
                {
                    SNESDestroy(&m_snes);
                    SNESCreate(PETSC_COMM_WORLD, &m_snes);
                }
                default_solver_configuration();
                m_is_set_up              = false;
                m_reuse_allocated_matrix = m_J != nullptr;
            }
        };

        template <class Scheme>
        class NonLinearSolver : public NonLinearSolverBase<Assembly<Scheme>>
        {
            using base_class = NonLinearSolverBase<Assembly<Scheme>>;

          public:

            using scheme_t       = Scheme;
            using input_field_t  = typename scheme_t::input_field_t;
            using output_field_t = typename scheme_t::output_field_t;
            using Mesh           = typename input_field_t::mesh_t;

            using base_class::assembly;
            using base_class::m_is_set_up;
            using base_class::m_J;
            using base_class::m_worker_output_field;

            explicit NonLinearSolver(const scheme_t& scheme)
                : base_class(scheme)
            {
            }

            void set_unknown(input_field_t& unknown)
            {
                assembly().set_unknown(unknown);
            }

            void solve(output_field_t& rhs)
            {
                m_worker_output_field = output_field_t("worker_output", rhs.mesh());

                if (!m_is_set_up)
                {
                    this->setup();
                }

                // update_ghost_mr(assembly().unknown());
                // update_ghost_mr(rhs);

                // samurai::save(fs::current_path(), "snes_initial_guess", {true, true}, assembly().unknown().mesh(), assembly().unknown());
                // samurai::save(fs::current_path(), "snes_rhs_before", {true, true}, rhs.mesh(), rhs);

                Vec b = assembly().create_rhs_vector(rhs);
                Vec x = assembly().create_solution_vector(assembly().unknown());

                // assembly().copy_rhs(b, rhs);
                // samurai::save(fs::current_path(), "snes_rhs_after", {true, true}, rhs.mesh(), rhs);

                this->prepare_rhs(x, b);

                this->solve_system(x, b);

#ifdef SAMURAI_WITH_MPI
                assembly().copy_unknown(x, assembly().unknown());
#endif
                VecDestroy(&b);
                VecDestroy(&x);
            }

            void solve(input_field_t& unknown, output_field_t& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

    } // end namespace petsc
} // end namespace samurai
