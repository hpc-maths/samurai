// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// A restart file written by some number of processes and read back by another.
// CTest runs the `write` test once on WRITER_PROCS processes, then the `read`
// test on several numbers of processes, fewer, as many and more (see
// CMakeLists.txt). Each file carries the expected global values as metadata.

#include <gtest/gtest.h>

#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

namespace
{
    constexpr std::size_t dim = 2;

    const samurai::Box<double, dim> box({0., 0.}, {1., 1.});

    auto mr_config()
    {
        return samurai::mesh_config<dim>().min_level(3).max_level(7).disable_minimal_ghost_width().disable_args_parse();
    }

    using uniform_mesh_t = samurai::UniformMesh<samurai::UniformConfig<dim>>;

    auto amr_config()
    {
        return samurai::mesh_config<dim>().min_level(5).max_level(5).disable_minimal_ghost_width().disable_args_parse();
    }

    // Linear in the coordinates, so that every cell can be checked exactly after
    // a restart, whatever process holds it.
    template <class Cell>
    double linear(const Cell& cell)
    {
        auto x = cell.center();
        return 1. + 2. * x[0] + 3. * x[1];
    }

    template <class Mesh>
    std::size_t global_cells(const Mesh& mesh)
    {
        mpi::communicator world;
        return mpi::all_reduce(world, mesh.nb_cells(Mesh::mesh_id_t::cells), std::plus<>());
    }

    template <class Field>
    double global_integral(const Field& u)
    {
        mpi::communicator world;
        double local = 0;
        samurai::for_each_cell(u.mesh(),
                               [&](const auto& cell)
                               {
                                   local += u[cell] * cell.length * cell.length;
                               });
        return mpi::all_reduce(world, local, std::plus<>());
    }

    template <class Mesh>
    void write_cell_count(const std::string& filename, const Mesh& mesh)
    {
        auto n_cells = global_cells(mesh);
        samurai::dump(
            filename,
            [&](auto& metadata)
            {
                metadata.attribute("n_cells", n_cells);
            },
            mesh);
    }

    template <class Mesh>
    void read_cell_count(const std::string& filename, Mesh& mesh)
    {
        std::size_t n_cells = 0;
        samurai::load(
            filename,
            [&](auto& metadata)
            {
                n_cells = metadata.template attribute<std::size_t>("n_cells");
            },
            mesh);
        EXPECT_TRUE_ALL_RANKS(global_cells(mesh) == n_cells);
    }

    // When the number of processes changed, load balances the cells: every
    // process gets some, and none gets much more than the average.
    template <class Mesh>
    void expect_balanced(const Mesh& mesh)
    {
        mpi::communicator world;
        auto local   = mesh.nb_cells(Mesh::mesh_id_t::cells);
        auto largest = mpi::all_reduce(world, local, mpi::maximum<std::size_t>());
        auto mean    = static_cast<double>(global_cells(mesh)) / world.size();
        EXPECT_TRUE_ALL_RANKS(local > 0);
        EXPECT_TRUE_ALL_RANKS(static_cast<double>(largest) <= 1.05 * mean);
    }

    std::size_t writer_procs(const std::string& filename)
    {
        HighFive::File file(filename + ".h5", HighFive::File::ReadOnly);
        return H5Easy::load<std::size_t>(file, "/n_process");
    }

    class RestartProcessCount : public samurai_test::MpiTest
    {
    };

    TEST_F(RestartProcessCount, write)
    {
        // MR mesh adapted to a disc, with a scalar and a vector field.
        auto mesh = samurai::mra::make_mesh(box, mr_config());
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   auto x  = cell.center();
                                   u[cell] = std::hypot(x[0] - 0.4, x[1] - 0.6) < 0.2 ? 1. : 0.;
                               });
        auto adapt      = samurai::make_MRAdapt(u);
        auto mra_config = samurai::mra_config().epsilon(1e-3);
        adapt(mra_config);

        auto v = samurai::make_vector_field<double, 2>("v", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   v[cell][0] = linear(cell);
                                   v[cell][1] = -linear(cell);
                               });

        auto n_cells  = global_cells(mesh);
        auto integral = global_integral(u);
        samurai::dump(
            "restart_mr",
            [&](auto& metadata)
            {
                metadata.attribute("n_cells", n_cells).attribute("integral", integral);
            },
            mesh,
            u,
            v);

        // The same mesh without any field.
        write_cell_count("restart_mr_mesh", mesh);

        // An AMR mesh.
        write_cell_count("restart_amr_mesh", samurai::amr::make_mesh(box, amr_config()));

        // A uniform mesh, which every process holds in full.
        uniform_mesh_t uniform(box, 4);
        auto s = samurai::make_scalar_field<double>("s", uniform);
        auto w = samurai::make_vector_field<int, 2>("w", uniform);
        samurai::for_each_cell(uniform,
                               [&](const auto& cell)
                               {
                                   s[cell]    = linear(cell);
                                   w[cell][0] = cell.indices[0];
                                   w[cell][1] = cell.indices[1];
                               });
        auto n_uniform_cells = uniform.nb_cells(uniform_mesh_t::mesh_id_t::cells);
        samurai::dump(
            "restart_uniform",
            [&](auto& metadata)
            {
                metadata.attribute("n_cells", n_uniform_cells);
            },
            uniform,
            s,
            w);
    }

    TEST_F(RestartProcessCount, read)
    {
        mpi::communicator world;
        bool changed = writer_procs("restart_mr") != static_cast<std::size_t>(world.size());

        {
            auto mesh = samurai::mra::make_empty_mesh(mr_config());
            auto u    = samurai::make_scalar_field<double>("u", mesh);
            auto v    = samurai::make_vector_field<double, 2>("v", mesh);

            std::size_t n_cells = 0;
            double integral     = 0;
            samurai::load(
                "restart_mr",
                [&](auto& metadata)
                {
                    n_cells  = metadata.template attribute<std::size_t>("n_cells");
                    integral = metadata.template attribute<double>("integral");
                },
                mesh,
                u,
                v);

            EXPECT_TRUE_ALL_RANKS(global_cells(mesh) == n_cells);
            EXPECT_TRUE_ALL_RANKS(std::abs(global_integral(u) - integral) <= 1e-12 * integral);

            bool exact = true;
            samurai::for_each_cell(mesh,
                                   [&](const auto& cell)
                                   {
                                       exact = exact && v[cell][0] == linear(cell) && v[cell][1] == -linear(cell);
                                   });
            EXPECT_TRUE_ALL_RANKS(exact);

            if (changed)
            {
                expect_balanced(mesh);
            }
        }
        {
            auto mesh = samurai::mra::make_empty_mesh(mr_config());
            read_cell_count("restart_mr_mesh", mesh);
            if (changed)
            {
                expect_balanced(mesh);
            }
        }
        {
            auto mesh = samurai::amr::make_empty_mesh(amr_config());
            read_cell_count("restart_amr_mesh", mesh);
            if (changed)
            {
                expect_balanced(mesh);
            }
        }
        {
            // Every process gets back the whole uniform mesh, whatever the
            // number of processes.
            uniform_mesh_t mesh;
            auto s = samurai::make_scalar_field<double>("s", mesh);
            auto w = samurai::make_vector_field<int, 2>("w", mesh);

            std::size_t n_cells = 0;
            samurai::load(
                "restart_uniform",
                [&](auto& metadata)
                {
                    n_cells = metadata.template attribute<std::size_t>("n_cells");
                },
                mesh,
                s,
                w);

            EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(uniform_mesh_t::mesh_id_t::cells) == n_cells);
            bool exact = true;
            samurai::for_each_cell(mesh,
                                   [&](const auto& cell)
                                   {
                                       exact = exact && s[cell] == linear(cell) && w[cell][0] == cell.indices[0]
                                            && w[cell][1] == cell.indices[1];
                                   });
            EXPECT_TRUE_ALL_RANKS(exact);
        }
    }
}
