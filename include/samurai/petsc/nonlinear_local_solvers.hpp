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
        template <class Scheme>
        class NonLinearLocalSolvers
        {
            // clang-format off

#define SAMURAI_STRINGIFY(x) #x
#define SAMURAI_VERSIONIFY(M, m, v) SAMURAI_STRINGIFY(M.m.v)


#ifdef SAMURAI_WITH_OPENMP
    #if !PetscDefined(HAVE_THREADSAFETY)
        #pragma message("To enable OpenMP for independent non-linear systems, PETSc must be configured with option --with-threadsafety.")
    #endif
    #if PETSC_VERSION_LT(3, 20, 6)
        #pragma message("To enable OpenMP for independent non-linear systems, upgrade PETSc to version 3.20.6 or upper (current version: " SAMURAI_VERSIONIFY(PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR) ")")
    #endif
    #if PetscDefined(HAVE_THREADSAFETY) && PETSC_VERSION_GE(3, 20, 6)
        #define ENABLE_PARALLEL_NONLINEAR_SOLVES
    #endif
#endif
            // clang-format on

            using scheme_t      = Scheme;
            using cfg_t         = typename scheme_t::cfg_t;
            using field_t       = typename scheme_t::field_t;
            using mesh_t        = typename field_t::mesh_t;
            using field_value_t = typename field_t::value_type;
            using cell_t        = Cell<mesh_t::dim, typename mesh_t::interval_t>;

          protected:

            field_t* m_unknown = nullptr;
            scheme_t m_scheme;

          public:

            // User callback to configure the solver
            std::function<void(SNES&, KSP&, PC&)> configure = nullptr;

            explicit NonLinearLocalSolvers(const scheme_t& scheme)
                : m_scheme(scheme)
            {
                if (!m_scheme.scheme_definition().local_scheme_function)
                {
                    std::cerr << "The scheme function 'local_scheme_function' of operator '" << scheme.name()
                              << "' has not been implemented." << std::endl;
                    assert(false && "Undefined 'local_scheme_function'");
                    exit(EXIT_FAILURE);
                }
                if (!m_scheme.scheme_definition().local_jacobian_function)
                {
                    std::cerr << "The function 'local_jacobian_function' of operator '" << scheme.name() << "' has not been implemented."
                              << std::endl;
                    assert(false && "Undefined 'local_jacobian_function'");
                    exit(EXIT_FAILURE);
                }
            }

          public:

            NonLinearLocalSolvers& operator=(const NonLinearLocalSolvers& other)
            {
                if (this != &other)
                {
                    this->m_unknown = other.m_unknown;
                }
                return *this;
            }

            NonLinearLocalSolvers& operator=(NonLinearLocalSolvers&& other)
            {
                if (this != &other)
                {
                    this->m_unknown = other.m_unknown;
                    other.m_unknown = nullptr;
                }
                return *this;
            }

            auto& scheme()
            {
                return m_scheme;
            }

            void set_unknown(field_t& u)
            {
                m_unknown = &u;
            }

            field_t& unknown()
            {
                return *m_unknown;
            }

          private:

            struct CellContextForPETSc
            {
                scheme_t* scheme;
                cell_t* cell;
                JacobianMatrix<cfg_t>* jac_coeffs;
                SchemeValue<cfg_t>* f_scheme_value;
            };

          public:

            void solve(field_t& rhs)
            {
                if (!m_unknown)
                {
                    std::cerr << "Undefined unknown for this non-linear system. Please set the unknowns using the instruction '[solver].set_unknown(u);'."
                              << std::endl;
                    assert(false && "Undefined unknown");
                    exit(EXIT_FAILURE);
                }
                static_assert(scheme_t::cfg_t::output_field_t::n_comp == field_t::n_comp);

                static constexpr PetscInt n = field_t::n_comp;

                times::timers.start("non-linear local solves");

#ifdef ENABLE_PARALLEL_NONLINEAR_SOLVES
                static constexpr Run run_type = Run::Parallel;
                std::size_t n_threads         = static_cast<std::size_t>(omp_get_max_threads());
#else
                static constexpr Run run_type = Run::Sequential;
                std::size_t n_threads         = 1;
#endif
                // Create workers
                std::vector<SNES> snes_list(n_threads);
                std::vector<JacobianMatrix<cfg_t>> J_coeffs_list(n_threads);
                std::vector<Mat> J_list(n_threads);
                std::vector<SchemeValue<cfg_t>> f_scheme_value_list(n_threads);
                std::vector<Vec> r_list(n_threads);
                std::vector<Vec> x_list(n_threads);
                std::vector<Vec> b_list(n_threads);

                // Allocate PETSc objects for each worker and configure the solvers
                for (std::size_t thread_num = 0; thread_num < n_threads; ++thread_num)
                {
                    SNESCreate(PETSC_COMM_SELF, &snes_list[thread_num]);
                    if (configure)
                    {
                        KSP ksp;
                        PC pc;
                        SNESGetKSP(snes_list[thread_num], &ksp);
                        KSPGetPC(ksp, &pc);
                        configure(snes_list[thread_num], ksp, pc);
                    }
                    SNESSetFromOptions(snes_list[thread_num]);
                    if constexpr (field_t::is_scalar)
                    {
                        MatCreateSeqDense(PETSC_COMM_SELF, n, n, &J_coeffs_list[thread_num], &J_list[thread_num]);
                    }
                    else
                    {
                        MatCreateSeqDense(PETSC_COMM_SELF, n, n, J_coeffs_list[thread_num].data(), &J_list[thread_num]);
                    }
                    VecCreateSeq(PETSC_COMM_SELF, n, &r_list[thread_num]);
                    VecCreateSeq(PETSC_COMM_SELF, n, &x_list[thread_num]);
                    VecCreateSeq(PETSC_COMM_SELF, n, &b_list[thread_num]);
                }

                for_each_cell<run_type>(unknown().mesh(),
                                        [&](auto& cell)
                                        {
#ifdef ENABLE_PARALLEL_NONLINEAR_SOLVES
                                            std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());
#else
                        std::size_t thread_num = 0;
#endif
                                            SNES& snes           = snes_list[thread_num];
                                            auto& J_coeffs       = J_coeffs_list[thread_num];
                                            Mat& J               = J_list[thread_num];
                                            Vec& r               = r_list[thread_num];
                                            auto& f_scheme_value = f_scheme_value_list[thread_num];
                                            Vec& x               = x_list[thread_num];
                                            Vec& b               = b_list[thread_num];

                                            copy(unknown(), cell, x); // local initial guess
                                            copy(rhs, cell, b);       // local right-hand side

                                            CellContextForPETSc ctx{&m_scheme, &cell, &J_coeffs, &f_scheme_value};
                                            SNESSetFunction(snes, r, PETSC_nonlinear_function, &ctx);
                                            SNESSetJacobian(snes, J, J, PETSC_jacobian_function, &ctx);
                                            // SNESSetFromOptions(snes);

                                            solve_system(snes, b, x); // solve the local non-linear system

                                            copy(x, unknown(), cell);
                                        });

                for (std::size_t thread_num = 0; thread_num < n_threads; ++thread_num)
                {
                    MatDestroy(&J_list[thread_num]);
                    VecDestroy(&r_list[thread_num]);
                    VecDestroy(&x_list[thread_num]);
                    VecDestroy(&b_list[thread_num]);
                    SNESDestroy(&snes_list[thread_num]);
                }

                times::timers.stop("non-linear local solves");
            }

          private:

            static PetscErrorCode PETSC_nonlinear_function(SNES, Vec x, Vec f, void* ctx)
            {
                auto petsc_ctx       = reinterpret_cast<CellContextForPETSc*>(ctx);
                auto& scheme         = *petsc_ctx->scheme;
                auto& cell           = *petsc_ctx->cell;
                auto& f_scheme_value = *petsc_ctx->f_scheme_value;

                // Wrap a LocalField structure around the data of the Petsc vector x
                const PetscScalar* x_data;
                VecGetArrayRead(x, &x_data);
                LocalField<field_t> x_field(cell, x_data);

                // Apply explicit scheme function to compute f = scheme(x)
                scheme.scheme_definition().local_scheme_function(f_scheme_value, cell, x_field);
                // Copy the computed f_scheme_value into the Petsc vector f
                copy(f_scheme_value, f);

                VecRestoreArrayRead(x, &x_data);
                return PETSC_SUCCESS;
            }

            static PetscErrorCode PETSC_jacobian_function(SNES, Vec x, Mat jac, Mat B, void* ctx)
            {
                // Here, jac = B = this.m_J

                auto petsc_ctx   = reinterpret_cast<CellContextForPETSc*>(ctx);
                auto& scheme     = *petsc_ctx->scheme;
                auto& cell       = *petsc_ctx->cell;
                auto& jac_coeffs = *petsc_ctx->jac_coeffs;

                // Wrap a LocalField structure around the data of the Petsc vector x
                const PetscScalar* x_data;
                VecGetArrayRead(x, &x_data);
                LocalField<field_t> x_field(cell, x_data);

                // Assembly of the Jacobian matrix.
                // In this case, jac = B, but Petsc recommends we assemble B for more general cases.

                // JacobianMatrix<cfg_t> jac_coeffs;

                scheme.scheme_definition().local_jacobian_function(jac_coeffs, cell, x_field);
                // No need to copy the Jacobian coefficients into the PETSc matrix,
                // since we provided the data pointer of jac_coeffs to the constructor of PETSc matrix so that they share the same memory.

                // if constexpr (field_t::is_scalar)
                // {
                //     MatSetValue(B, 0, 0, jac_coeffs, INSERT_VALUES);
                // }
                // else
                // {
                // for (PetscInt i = 0; i < static_cast<PetscInt>(field_t::n_comp); ++i)
                // {
                //     for (PetscInt j = 0; j < static_cast<PetscInt>(field_t::n_comp); ++j)
                //     {
                //         MatSetValue(B, i, j, jac_coeffs(i, j), INSERT_VALUES);
                //     }
                // }
                // std::array<PetscInt, field_t::n_comp> rows;
                // for (std::size_t i = 0; i < field_t::n_comp; ++i)
                // {
                //     rows[i] = static_cast<PetscInt>(i);
                // }
                // MatSetValues(B,
                //              static_cast<PetscInt>(field_t::n_comp),
                //              rows.data(),
                //              static_cast<PetscInt>(field_t::n_comp),
                //              rows.data(),
                //              jac_coeffs.data(),
                //              INSERT_VALUES);
                // }
                PetscObjectSetName(reinterpret_cast<PetscObject>(B), "Jacobian");
                MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
                if (jac != B)
                {
                    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
                }

                // MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                VecRestoreArrayRead(x, &x_data);

                return PETSC_SUCCESS;
            }

          protected:

            void solve_system(SNES snes, Vec& b, Vec& x)
            {
                // Solve the system
                SNESSolve(snes, b, x);

                SNESConvergedReason reason_code;
                SNESGetConvergedReason(snes, &reason_code);
                if (reason_code < 0)
                {
                    using namespace std::string_literals;
                    const char* reason_text;
                    SNESGetConvergedReasonString(snes, &reason_text);
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

            void solve(field_t& unknown, field_t& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }
        };

    } // end namespace petsc
} // end namespace samurai
