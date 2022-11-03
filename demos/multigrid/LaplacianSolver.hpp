#pragma once
#include "samurai_new/multigrid/petsc/GeometricMultigrid.hpp"
#include "utils.hpp"

template<class Dsctzr>
class LaplacianSolver
{
    using Mesh = typename Dsctzr::Mesh;
    using Field = typename Dsctzr::field_t;

private:
    Dsctzr& _discretizer;
    KSP _ksp;
    bool _use_samurai_mg;
    samurai_new::petsc::GeometricMultigrid<Dsctzr> _samurai_mg;


public:
    LaplacianSolver(Dsctzr& discretizer, Mesh& mesh)
    : _discretizer(discretizer)
    {
        create_solver(discretizer, mesh);
    }

    void destroy_petsc_objects()
    {
        _samurai_mg.destroy_petsc_objects();
        KSPDestroy(&_ksp);
    }

private:
    void create_solver(Dsctzr& discretizer, Mesh& mesh)
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
            _samurai_mg = samurai_new::petsc::GeometricMultigrid(discretizer, mesh);
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

    void solve(const Vec& b, Field& x_field)
    {
        Vec x;
        VecDuplicate(b, &x);
        
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

        samurai_new::petsc::copy(x, x_field);
        VecDestroy(&x);
    }
};