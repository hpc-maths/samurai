#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

// A peak near the bottom left corner: the refined cells gather there
double peak(double x, double y)
{
    const double r2 = (x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3);
    return std::exp(-200. * r2);
}

int main(int argc, char** argv)
{
    samurai::initialize("Adapt a mesh with MPI", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;
    using mesh_id_t                  = samurai::MRMeshId;

    // Each process gets a part of the mesh
    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto cfg  = samurai::mesh_config<dim>().min_level(3).max_level(7);
    auto mesh = samurai::mra::make_mesh(box, cfg);

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto init = [&](const auto& cell)
    {
        u[cell] = peak(cell.center(0), cell.center(1));
    };
    samurai::for_each_cell(mesh, init);
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(samurai::mra_config().epsilon(1e-3));

    // The cells of this process and the integral of u on them
    std::size_t n_local = mesh.nb_cells(mesh_id_t::cells);
    double sum_local    = 0.;
    auto add_cell       = [&](const auto& cell)
    {
        sum_local += u[cell] * cell.length * cell.length;
    };
    samurai::for_each_cell(mesh, add_cell);

    int rank = 0;
    int size = 1;
    std::vector<std::size_t> counts{n_local};
    std::size_t total = n_local;
    double integral   = sum_local;
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    rank = world.rank();
    size = world.size();
    // Rank 0 collects the number of cells of every process
    mpi::gather(world, n_local, counts, 0);
    // Every process gets the global values
    total    = mpi::all_reduce(world, n_local, std::plus<>());
    integral = mpi::all_reduce(world, sum_local, std::plus<>());
#endif

    if (rank == 0)
    {
        for (std::size_t r = 0; r < counts.size(); ++r)
        {
            std::cout << "rank " << r << ": " << counts[r];
            std::cout << " cells\n";
        }
        std::cout << "total: " << total << " cells\n";
        std::cout << fmt::format("integral of u: {:.15e}\n", integral);
    }

    // Every process writes its cells into the same file
    samurai::save(fmt::format("mpi_adapt_size_{}", size), mesh, u);

    // Only rank 0 prints this line, unless --dont-redirect-output
    std::cout << "rank " << rank << " saved " << n_local << " cells\n";

    samurai::finalize();
    return 0;
}
