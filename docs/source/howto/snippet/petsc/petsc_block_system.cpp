#include <cmath>
#include <iostream>

#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

void solve_coupled_heat()
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
    auto v    = samurai::make_scalar_field<double>("v", mesh, 0.);
    auto unp1 = samurai::make_scalar_field<double>("u", mesh, 0.);
    auto vnp1 = samurai::make_scalar_field<double>("v", mesh, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);
    samurai::make_bc<samurai::Neumann<1>>(vnp1, 0.);

    // u' = lap(u) - k (u - v)  and  v' = lap(v) + k (u - v)
    double dt = 1e-3;
    double k  = 10;
    auto diff = samurai::make_diffusion_order2<decltype(u)>();
    auto id   = samurai::make_identity<decltype(u)>();
    auto Aii  = (1 + dt * k) * id + dt * diff;
    auto Aij  = (-dt * k) * id;

    auto op = samurai::make_block_operator<2, 2>(Aii, Aij, Aij, Aii);

    // one PETSc matrix per block
    using enum samurai::petsc::BlockAssemblyType;
    auto solver = samurai::petsc::make_solver<NestedMatrices>(op);
    solver.set_unknowns(unp1, vnp1);
    solver.after_matrix_assembly = [&](KSP&, PC& pc, Mat&)
    {
        solver.set_pc_fieldsplit(pc); // one split per unknown
    };

    for (int n = 1; n <= 3; ++n)
    {
        solver.solve(u, v); // one right-hand side per block row

        int its = solver.iterations();
        std::cout << "step " << n << ": " << its << " iterations\n";

        samurai::swap(u, unp1);
        samurai::swap(v, vnp1);
    } // end of the time loop
}

int main(int argc, char** argv)
{
    samurai::initialize("Implicit coupled heat equations", argc, argv);
    SAMURAI_PARSE(argc, argv);

    solve_coupled_heat(); // its solver is destroyed on return

    samurai::finalize();
    return 0;
}
