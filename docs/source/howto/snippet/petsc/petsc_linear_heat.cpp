#include <algorithm>
#include <cmath>
#include <iostream>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

// This example needs PETSc: it does nothing in a build without it.
#ifdef SAMURAI_WITH_PETSC

// Largest value of |f - g| over the cells of all ranks
template <class Field>
double max_diff(const Field& f, const Field& g)
{
    double result = 0;
    auto update   = [&](const auto& cell)
    {
        result = std::max(result, std::abs(f[cell] - g[cell]));
    };
    samurai::for_each_cell(f.mesh(), update);
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    result = mpi::all_reduce(world, result, mpi::maximum<double>());
#endif
    return result;
}

void solve_heat()
{
    samurai::Box<double, 2> box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<2>().min_level(6).max_level(6);
    auto mesh   = samurai::mra::make_mesh(box, config);

    auto bump = [](const auto& x)
    {
        double r2 = std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2);
        return std::exp(-100 * r2);
    };
    auto u    = samurai::make_scalar_field<double>("u", mesh, bump);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

    double dt = 1e-3;
    auto diff = samurai::make_diffusion_order2<decltype(u)>();
    auto id   = samurai::make_identity<decltype(u)>();
    auto A    = id + dt * diff;

    auto solver = samurai::petsc::make_solver(A);
    solver.set_unknown(unp1);
    solver.configure = [](KSP& ksp, PC& pc)
    {
        KSPSetType(ksp, KSPPREONLY); // -ksp_type preonly
        PCSetType(pc, PCLU);         // -pc_type lu
    };

    for (int n = 1; n <= 3; ++n)
    {
        solver.solve(u); // solves A(unp1) = u

        auto Au  = A(unp1);
        double r = max_diff(u, Au);
        int its  = solver.iterations();
        std::cout << "step " << n << ": " << its << " iterations";
        std::cout << ", residual " << r << std::endl;

        samurai::swap(u, unp1);
    }
}

int main(int argc, char** argv)
{
    samurai::initialize("Implicit heat equation", argc, argv);
    SAMURAI_PARSE(argc, argv);

    solve_heat(); // its solver is destroyed on return

    samurai::finalize();
    return 0;
}

#else

int main()
{
    std::cout << "This example needs PETSc." << std::endl;
    return 0;
}

#endif
