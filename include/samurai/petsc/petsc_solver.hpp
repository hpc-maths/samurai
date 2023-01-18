#pragma once

//#define ENABLE_MG

#ifdef ENABLE_MG
#include "multigrid/petsc/GeometricMultigrid.hpp"
#else
#include "utils.hpp"
#endif

namespace samurai { namespace petsc
{
    template<class Dsctzr>
    class PetscSolver
    {
        using Mesh = typename Dsctzr::Mesh;
        using Field = typename Dsctzr::field_t;

    private:
        const Dsctzr& _discretizer;
        KSP _ksp = nullptr;
        bool _use_samurai_mg = false;
        Mat _A = nullptr;
        bool _is_set_up = false;
#ifdef ENABLE_MG
        GeometricMultigrid<Dsctzr> _samurai_mg;
#endif


    public:
        PetscSolver(const Dsctzr& discretizer)
        : _discretizer(discretizer)
        {
            create_solver(_discretizer.mesh());
        }

        ~PetscSolver()
        {
            destroy_petsc_objects();
        }

        void destroy_petsc_objects()
        {
#ifdef ENABLE_MG
            _samurai_mg.destroy_petsc_objects();
#endif
            if (_A)
            {
                MatDestroy(&_A);
                _A = nullptr;
            }
            /*if (_ksp)
            {
                KSPDestroy(&_ksp);
                _ksp = nullptr;
            }*/
        }

    private:
        void create_solver(Mesh&
#ifdef ENABLE_MG
        mesh
#endif
        )
        {
            KSP user_ksp;
            KSPCreate(PETSC_COMM_SELF, &user_ksp);
            KSPSetFromOptions(user_ksp);
            PC user_pc;
            KSPGetPC(user_ksp, &user_pc);
            PCType user_pc_type;
            PCGetType(user_pc, &user_pc_type);
#ifdef ENABLE_MG
            _use_samurai_mg = strcmp(user_pc_type, PCMG) == 0;
#endif
            KSPDestroy(&user_ksp);

            KSPCreate(PETSC_COMM_SELF, &_ksp);
            KSPSetFromOptions(_ksp);
#ifdef ENABLE_MG
            if (_use_samurai_mg)
            {
                if constexpr(Mesh::dim > 2)
                {
                    std::cerr << "Samurai Multigrid is not implemented for dim > 2." << std::endl;
                    assert(false);
                    exit(EXIT_FAILURE);
                }
                _samurai_mg = GeometricMultigrid(_discretizer, mesh);
                _samurai_mg.apply_as_pc(_ksp);
            }
#endif
        }

    public:
        void setup()
        {
            if (_is_set_up)
            {
                return;
            }
            if (!_use_samurai_mg)
            {
                _discretizer.create_matrix(_A);
                _discretizer.assemble_matrix(_A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(_A), "A");
                KSPSetOperators(_ksp, _A, _A);
            }
            KSPSetUp(_ksp);
            _is_set_up = true;
        }

        void solve(const Field& source, Field& solution)
        {
            if (!_is_set_up)
            {
                setup();
            }

            // Create right-hand side vector from the source field
            Vec b = samurai::petsc::create_petsc_vector_from(source);
            PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b"); //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Update the right-hand side with the boundary conditions stored in the solution field
            _discretizer.enforce_bc(b, solution);                      //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Create the solution vector
            Vec x;
            VecDuplicate(b, &x);

            // Solve the system
            KSPSolve(_ksp, b, x);

            KSPConvergedReason reason_code;
            KSPGetConvergedReason(_ksp, &reason_code);
            if (reason_code < 0)
            {
                using namespace std::string_literals;
                const char* reason_text;
                KSPGetConvergedReasonString(_ksp, &reason_text);
                std::cerr << "Divergence of the solver ("s + reason_text + ")" << std::endl;
                assert(false);
                exit(EXIT_FAILURE);
            }
            //VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            VecDestroy(&b);
            copy(x, solution);
            VecDestroy(&x);
        }

        int iterations()
        {
            PetscInt n_iterations;
            KSPGetIterationNumber(_ksp, &n_iterations);
            return n_iterations;
        }
    };

    template<class Dsctzr>
    void solve(const Dsctzr& discretizer, const typename Dsctzr::field_t& rhs, typename Dsctzr::field_t& solution)
    {
        PetscSolver<Dsctzr> solver(discretizer);
        solver.solve(rhs, solution);
    }

}} // end namespace