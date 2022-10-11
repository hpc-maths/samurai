#include <iostream>
#include <fstream>

#include <fmt/format.h>

#include <boost/mpi.hpp>
#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/field.hpp>
#include <samurai/algorithm/update.hpp>

namespace mpi = boost::mpi;
int main()
{
    constexpr std::size_t dim = 1;
    mpi::environment env;
    mpi::communicator world;

    // // Test interval
    // if (world.rank() == 0)
    // {
    //     mpi::request reqs[2];
    //     samurai::Interval<int> i_0 = {0, 10}, i_1;
    //     reqs[0] = world.isend(1, 0, i_0);
    //     reqs[1] = world.irecv(1, 1, i_1);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << i_1 << "!" << std::endl;
    // }
    // else
    // {
    //     mpi::request reqs[2];
    //     samurai::Interval<int> i_0, i_1 = {10, 20};
    //     reqs[0] = world.isend(0, 1, i_1);
    //     reqs[1] = world.irecv(0, 0, i_0);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << i_0 << ", " << std::endl;
    // }

    // // Test CellArray
    // if (world.rank() == 0)
    // {
    //     mpi::request reqs[2];
    //     samurai::Box<int, 1> box = {{0}, {10}};
    //     samurai::LevelCellArray<1> lca_0{0, box}, lca_1;
    //     samurai::CellArray<1> ca_0, ca_1;
    //     ca_0[0] = lca_0;
    //     reqs[0] = world.isend(1, 0, ca_0);
    //     reqs[1] = world.irecv(1, 1, ca_1);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << ca_1 << "!" << std::endl;
    // }
    // else
    // {
    //     mpi::request reqs[2];
    //     samurai::Box<int, 1> box = {{10}, {20}};
    //     samurai::LevelCellArray<1> lca_0, lca_1{0, box};
    //     samurai::CellArray<1> ca_0, ca_1;
    //     ca_1[0] = lca_1;
    //     reqs[0] = world.isend(0, 1, ca_1);
    //     reqs[1] = world.irecv(0, 0, ca_0);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << ca_0 << ", " << std::endl;
    // }

    samurai::Box<double, dim> box = {{0}, {1}};
    using Config = samurai::MRConfig<dim>;
    samurai::MRMesh<Config> mesh{box, 2, 4};

    std::ofstream out(fmt::format("output_{}.txt", world.rank()));
    out << mesh << std::endl;

    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(world.rank());
    samurai::update_ghost_mpi(u);
    out << u << std::endl;
    // // Test MRMesh
    // if (world.rank() == 0)
    // {
    //     mpi::request reqs[2];
    //     samurai::Box<double, dim> box = {{0}, {1}};
    //     using Config = samurai::MRConfig<dim>;
    //     samurai::MRMesh<Config> mesh_0{box, 2, 4}, mesh_1;
    //     reqs[0] = world.isend(1, 0, mesh_0);
    //     reqs[1] = world.irecv(1, 1, mesh_1);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << mesh_1 << "!" << std::endl;
    // }
    // else
    // {
    //     mpi::request reqs[2];
    //     samurai::Box<double, 1> box = {{1}, {2}};
    //     using Config = samurai::MRConfig<dim>;
    //     samurai::MRMesh<Config> mesh_0, mesh_1{box, 2, 4};
    //     reqs[0] = world.isend(0, 1, mesh_1);
    //     reqs[1] = world.irecv(0, 0, mesh_0);
    //     mpi::wait_all(reqs, reqs + 2);
    //     std::cout << world.rank() << mesh_0 << ", " << std::endl;
    // }

    return 0;
}