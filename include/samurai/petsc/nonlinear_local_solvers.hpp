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
            using scheme_t      = Scheme;
            using field_t       = typename scheme_t::field_t;
            using mesh_t        = typename field_t::mesh_t;
            using field_value_t = typename field_t::value_type;
            using cell_t        = Cell<mesh_t::dim, typename mesh_t::interval_t>;

          protected:

            field_t* m_unknown = nullptr;
            scheme_t m_scheme;
            Mat m_J = nullptr;

            bool m_is_set_up = false;

          public:

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

                //_configure_solver(m_snes);
            }

            virtual ~NonLinearLocalSolvers()
            {
                _destroy_petsc_objects();
            }

          private:

            void _destroy_petsc_objects()
            {
                // if (m_J)
                // {
                //     MatDestroy(&m_J);
                //     m_J = nullptr;
                // }
                // if (m_snes)
                // {
                //     SNESDestroy(&m_snes);
                //     m_snes = nullptr;
                // }
            }

          public:

            virtual void destroy_petsc_objects()
            {
                _destroy_petsc_objects();
            }

            NonLinearLocalSolvers& operator=(const NonLinearLocalSolvers& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_unknown = other.m_unknown;
                    // this->m_snes      = other.m_snes;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                }
                return *this;
            }

            NonLinearLocalSolvers& operator=(NonLinearLocalSolvers&& other)
            {
                if (this != &other)
                {
                    this->destroy_petsc_objects();
                    this->m_unknown = other.m_unknown;
                    // this->m_snes      = other.m_snes;
                    this->m_J         = other.m_J;
                    this->m_is_set_up = other.m_is_set_up;
                    other.m_unknown   = nullptr;
                    // other.m_snes      = nullptr; // Prevent SNES destruction when 'other' object is destroyed
                    other.m_J         = nullptr;
                    other.m_is_set_up = false;
                }
                return *this;
            }

            bool is_set_up()
            {
                return m_is_set_up;
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

          protected:

            // virtual void configure_solver()
            // {
            //     _configure_solver();
            // }

          private:

            struct CellContextForPETSc
            {
                scheme_t* scheme;
                cell_t* cell;
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
                static_assert(scheme_t::cfg_t::output_field_size == field_t::size);

                static constexpr PetscInt n = field_t::size;
                // MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, &m_J);

#ifdef SAMURAI_WITH_OPENMP
                std::size_t n_threads = static_cast<std::size_t>(omp_get_max_threads());
#else
                std::size_t n_threads = 1;
#endif
                // std::vector<std::array<field_value_t, n>> x_data_list(n_threads);
                std::vector<SNES> snes_list(n_threads);
                // std::vector<CellContextForPETSc> context_list(n_threads);
                std::vector<Mat> J_list(n_threads);
                // std::vector<Vec> b_list(n_threads);
                // std::vector<Vec> x_list(n_threads);

                // #pragma omp parallel // for
                for (std::size_t thread_num = 0; thread_num < n_threads; ++thread_num)
                {
                    // std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());

                    SNESCreate(PETSC_COMM_SELF, &snes_list[thread_num]);

                    // context_list.push_back({&m_scheme, nullptr});

                    // Non-linear function
                    // SNESSetFunction(snes, nullptr, PETSC_nonlinear_function, &context_list[i]);
                    // std::cout << "[" << i << "] configuration snes = " << snes << ", ctx = " << &context_list[i] << std::endl;
                    // // Jacobian matrix
                    // SNESSetJacobian(snes, m_J, m_J, PETSC_jacobian_function, &context_list[i]);

                    MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, &J_list[thread_num]);

                    // Vec b;
                    // VecCreateSeq(PETSC_COMM_SELF, n, &b_list[thread_num]);
                    // Vec x;
                    // VecCreateSeq(PETSC_COMM_SELF, n, &x_list[thread_num]);
                    // VecCreateSeqWithArray(MPI_COMM_SELF, 1, n, PETSC_NULL_SCALAR, &x_list[thread_num]);
                }

                parallel_for_each_cell(unknown().mesh(),
                                       [&](auto& cell)
                                       {
                                           std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());

                                           // std::cout << "[" << thread_num << "] level " << cell.level << ", cell " << cell.index <<
                                           // std::endl;

                                           SNES& snes = snes_list[thread_num];
                                           Mat& J     = J_list[thread_num];
                                           Vec x;
                                           Vec b;

                                           if constexpr (n > 1 && field_t::is_soa)
                                           {
                                               VecCreateSeq(PETSC_COMM_SELF, n, &x);
                                               copy(unknown(), cell, x);

                                               VecCreateSeq(PETSC_COMM_SELF, n, &b);
                                               copy(rhs, cell, b);
                                           }
                                           else
                                           {
                                               x = create_petsc_vector_from(unknown(), cell);
                                               b = create_petsc_vector_from(rhs, cell);
                                           }

                                           // CellContextForPETSc& ctx = context_list[thread_num];
                                           // ctx.cell                 = &cell;
                                           CellContextForPETSc ctx{&m_scheme, &cell};

                                           // Non-linear function
                                           // Vec r;
                                           // VecCreateSeq(PETSC_COMM_SELF, n, &r);
                                           SNESSetFunction(snes, nullptr, PETSC_nonlinear_function, &ctx);
                                           // Jacobian matrix
                                           SNESSetJacobian(snes, J, J, PETSC_jacobian_function, &ctx);

                                           SNESSetFromOptions(snes);

                                           solve_system(snes, b, x);

                                           if constexpr (n > 1 && field_t::is_soa)
                                           {
                                               copy(x, unknown(), cell);
                                           }

                                           VecDestroy(&x);
                                           VecDestroy(&b);
                                       });

                // #pragma omp parallel // for
                for (std::size_t thread_num = 0; thread_num < n_threads; ++thread_num)
                {
                    // std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());
                    MatDestroy(&J_list[thread_num]);
                    SNESDestroy(&snes_list[thread_num]);
                    // VecDestroy(&b_list[thread_num]);
                    // VecDestroy(&x_list[thread_num]);
                }
            }

          private:

            static PetscErrorCode PETSC_nonlinear_function(SNES /*snes*/, Vec x, Vec f, void* ctx)
            {
                // std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());
                //  std::cout << "[" << thread_num << "] PETSC_nonlinear_function: snes = " << snes << ", ctx=" << ctx << std::endl;

                CellContextForPETSc* petsc_ctx = reinterpret_cast<CellContextForPETSc*>(ctx);
                auto& scheme                   = *petsc_ctx->scheme;
                auto& cell                     = *petsc_ctx->cell;

                // Wrap a LocalField structure around the data of the Petsc vector x
                const PetscScalar* x_data;
                // #pragma omp critical
                //               {
                VecGetArrayRead(x, &x_data);
                LocalField<field_t> x_field(cell, x_data);
                VecRestoreArrayRead(x, &x_data);

                // PetscScalar* f_data;
                // VecGetArray(f, &f_data);
                // LocalField<field_t> f_field(cell, f_data);

                // Apply explicit scheme
                auto f_field = scheme.scheme_definition().local_scheme_function(cell, x_field);

                copy(f_field, f);
                //          }

                // VecRestoreArrayRead(x, &x_data);
                //  VecRestoreArray(f, &f_data);
                return 0; // PETSC_SUCCESS
            }

            static PetscErrorCode PETSC_jacobian_function(SNES /*snes*/, Vec x, Mat jac, Mat B, void* ctx)
            {
                // Here, jac = B = this.m_J

                auto petsc_ctx = reinterpret_cast<CellContextForPETSc*>(ctx);
                auto& scheme   = *petsc_ctx->scheme;
                auto& cell     = *petsc_ctx->cell;

                // Wrap a LocalField structure around the data of the Petsc vector x
                const PetscScalar* x_data;
                VecGetArrayRead(x, &x_data);
                LocalField<field_t> x_field(cell, x_data);

                // Assembly of the Jacobian matrix.
                // In this case, jac = B, but Petsc recommends we assemble B for more general cases.
                auto jac_stencil_coeffs = scheme.scheme_definition().local_jacobian_function(cell, x_field);
                auto& jac_coeffs        = jac_stencil_coeffs[0]; // local stencil (of size 1)
                if constexpr (field_t::size == 1)
                {
                    MatSetValue(B, 0, 0, jac_coeffs, INSERT_VALUES);
                }
                else
                {
                    for (PetscInt i = 0; i < static_cast<PetscInt>(field_t::size); ++i)
                    {
                        for (PetscInt j = 0; j < static_cast<PetscInt>(field_t::size); ++j)
                        {
                            MatSetValue(B, i, j, jac_coeffs(i, j), INSERT_VALUES);
                        }
                    }
                }
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

                return 0; // PETSC_SUCCESS
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

            virtual void reset()
            {
                destroy_petsc_objects();
                m_is_set_up = false;
            }
        };

    } // end namespace petsc
} // end namespace samurai
