#pragma once
#include "multigrid/petsc/GeometricMultigrid.hpp"
#include "../utils.hpp"

namespace samurai_new { namespace petsc
{
    template<class Dsctzr>
    class PetscDiffusionSolver
    {
        using Mesh = typename Dsctzr::Mesh;
        using Field = typename Dsctzr::field_t;
        using boundary_condition_t = typename Field::boundary_condition_t;

    private:
        Dsctzr _discretizer;
        KSP _ksp;
        bool _use_samurai_mg;
        GeometricMultigrid<Dsctzr> _samurai_mg;


    public:
        PetscDiffusionSolver(Mesh& mesh, const std::vector<boundary_condition_t>& boundary_conditions)
        : _discretizer(mesh, boundary_conditions)
        {
            create_solver(mesh);
        }

        void destroy_petsc_objects()
        {
            _samurai_mg.destroy_petsc_objects();
            KSPDestroy(&_ksp);
        }

    private:
        void create_solver(Mesh& mesh)
        {
            KSP user_ksp;
            KSPCreate(PETSC_COMM_SELF, &user_ksp);
            KSPSetFromOptions(user_ksp);
            PC user_pc;
            KSPGetPC(user_ksp, &user_pc);
            PCType user_pc_type;
            PCGetType(user_pc, &user_pc_type);
            _use_samurai_mg = strcmp(user_pc_type, PCMG) == 0;
            KSPDestroy(&user_ksp);

            KSPCreate(PETSC_COMM_SELF, &_ksp);
            KSPSetFromOptions(_ksp);
            if (_use_samurai_mg)
            {
                if constexpr(Mesh::dim > 2)
                {
                    fatal_error("Samurai Multigrid is not implemented for dim > 2.");
                }
                _samurai_mg = GeometricMultigrid(_discretizer, mesh);
                _samurai_mg.apply_as_pc(_ksp);
            }
        }

    public:
        void setup()
        {
            if (!_use_samurai_mg)
            {
                Mat A;
                _discretizer.create_matrix(A);
                _discretizer.assemble_matrix(A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(A), "A");
                KSPSetOperators(_ksp, A, A);
            }
            KSPSetUp(_ksp);
        }

        void solve(const Field& source, Field& solution)
        {
            // Create right-hand side vector from the source field
            Vec b = samurai::petsc::create_petsc_vector_from(source);
            PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b");

            // Update the right-hand side with the boundary conditions stored in the solution field
            _discretizer.enforce_bc(b, solution);

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
                fatal_error("Divergence of the solver ("s + reason_text + ")");
            }

            PetscInt n_iterations;
            KSPGetIterationNumber(_ksp, &n_iterations);
            std::cout << n_iterations << " iterations" << std::endl;
            std::cout << std::endl;
            //VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            VecDestroy(&b);
            samurai::petsc::copy(x, solution);
            VecDestroy(&x);
        }
    };
}} // end namespace