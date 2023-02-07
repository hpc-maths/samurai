#pragma once

//#define ENABLE_MG

#include "petsc_block_assembly.hpp"
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
        Dsctzr& _discretizer;
        KSP _ksp = nullptr;
        bool _use_samurai_mg = false;
        Mat _A = nullptr;
        bool _is_set_up = false;
#ifdef ENABLE_MG
        GeometricMultigrid<Dsctzr> _samurai_mg;
#endif


    public:
        PetscSolver(Dsctzr& discretizer)
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

        void solve(const Field& source)
        {
            if (!_is_set_up)
            {
                setup();
            }

            // Create right-hand side vector from the source field
            Vec b = create_petsc_vector_from(source);
            PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b"); //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Update the right-hand side with the boundary conditions stored in the solution field
            _discretizer.enforce_bc(b);                      //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Create the solution vector
            Vec x = create_petsc_vector_from(_discretizer.unknown());

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
    PetscSolver<Dsctzr> make_solver(Dsctzr& discretizer)
    {
        return PetscSolver<Dsctzr>(discretizer);
    }


    template<class Dsctzr>
    void solve(Dsctzr& discretizer, const typename Dsctzr::field_t& rhs)
    {
        PetscSolver<Dsctzr> solver(discretizer);
        solver.solve(rhs);
    }



    /**
     * PETSc block solver
    */
    template <int rows, int cols, class... Operators>
    class PetscSolver<PetscBlockAssembly<rows, cols, Operators...>>
    {
        using Dsctzr = PetscBlockAssembly<rows, cols, Operators...>;
    private:
        Dsctzr& _discretizer;
        KSP _ksp = nullptr;
        Mat _A = nullptr;
        bool _is_set_up = false;
    public:
        PetscSolver(Dsctzr& discretizer)
        : _discretizer(discretizer)
        {
            create_solver();
        }

        ~PetscSolver()
        {
            destroy_petsc_objects();
        }

        void destroy_petsc_objects()
        {
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

        KSP& Ksp()
        {
            return _ksp;
        }

    private:
        void create_solver()
        {
            KSPCreate(PETSC_COMM_SELF, &_ksp);
            //KSPSetFromOptions(_ksp);
        }

    public:
        void setup()
        {
            if (_is_set_up)
            {
                return;
            }
            KSPSetFromOptions(_ksp);

            _discretizer.create_matrix(_A);
            _discretizer.assemble_matrix(_A);
            PetscObjectSetName(reinterpret_cast<PetscObject>(_A), "A"); //MatView(_A, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
            KSPSetOperators(_ksp, _A, _A);



            // Set names to the petsc fields
            PC pc;
            KSPGetPC(_ksp,&pc);
            IS is_fields[cols];
            MatNestGetISs(_A, is_fields, NULL);
            auto field_names = _discretizer.field_names();
            for (std::size_t i=0; i<cols; ++i)
            {
                PCFieldSplitSetIS(pc, field_names[i].c_str(), is_fields[i]);
            }

            //KSPSetUp(_ksp); // TO UNCOMMENT!!!!!!!!
            _is_set_up = true;
        }

        template<class... Fields>
        void solve(const std::tuple<Fields&...>& sources)
        {
            static_assert(sizeof...(Fields) == rows, "The number of source fields passed to solve() must equal the number of rows of the block operator.");

            if (!_is_set_up)
            {
                setup();
            }

            // Create a right-hand side block-vector from the source fields
            std::array<Vec, rows> b_blocks;
            std::size_t i = 0;
            for_each(sources, [&](auto& s) 
            {
                b_blocks[i] = create_petsc_vector_from(s);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b_blocks[i]), s.name().c_str());
                i++;
            });
            Vec b;
            VecCreateNest(PETSC_COMM_SELF, rows, NULL, b_blocks.data(), &b);
            PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side"); //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Update the right-hand side with the boundary conditions stored in the solution field
            _discretizer.enforce_bc(b_blocks);               //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

            // Create the solution vector
            std::array<Vec, cols> x_blocks = _discretizer.create_solution_vectors();
            Vec x;
            VecCreateNest(PETSC_COMM_SELF, cols, NULL, x_blocks.data(), &x);
            PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution"); //VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

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
            VecDestroy(&x);
        }

        int iterations()
        {
            PetscInt n_iterations;
            KSPGetIterationNumber(_ksp, &n_iterations);
            return n_iterations;
        }
    };

    template <int rows, int cols>
    PetscSolver<PetscBlockAssembly<rows, cols>> make_solver(PetscBlockAssembly<rows, cols>& discretizer)
    {
        return PetscSolver<PetscBlockAssembly<rows, cols>>(discretizer);
    }

}} // end namespace