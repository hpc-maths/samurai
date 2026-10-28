// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Sub-mesh options passed to save and local_save as a braced list, such as
// `{true}` or `{true, true}`, must build an Hdf5Options and never a
// communicator: with MPICH, MPI_Comm is an int and `{true}` converts to it.
// Each overload that accepts options is called, and the test checks that the
// written file holds the grids the options ask for.

#include <filesystem>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

#include "mpi_test_utils.hpp"

namespace fs  = std::filesystem;
namespace mpi = boost::mpi;

namespace
{
    constexpr std::size_t dim   = 2;
    constexpr std::size_t level = 4;

    enum class Overload
    {
        path,
        filename,
        path_comm,
        filename_comm,
        path_boost_comm,
        local_path,
        local_filename
    };

    std::string overload_name(Overload overload)
    {
        switch (overload)
        {
            case Overload::path:
                return "path";
            case Overload::filename:
                return "filename";
            case Overload::path_comm:
                return "path_comm";
            case Overload::filename_comm:
                return "filename_comm";
            case Overload::path_boost_comm:
                return "path_boost_comm";
            case Overload::local_path:
                return "local_path";
            case Overload::local_filename:
                return "local_filename";
        }
        return "unknown";
    }

    // Lets GoogleTest print the overload by name in failure messages.
    std::ostream& operator<<(std::ostream& os, Overload overload)
    {
        return os << overload_name(overload);
    }

    bool is_local(Overload overload)
    {
        return overload == Overload::local_path || overload == Overload::local_filename;
    }

    bool has_path(Overload overload)
    {
        return overload == Overload::path || overload == Overload::path_comm || overload == Overload::path_boost_comm
            || overload == Overload::local_path;
    }

    class SaveOptions : public samurai_test::MpiTest,
                        public ::testing::WithParamInterface<Overload>
    {
      protected:

        // The tests for 2, 3 and 4 processes may run at the same time in the
        // same working directory, so the file names carry the process count.
        std::string filename(const std::string& scenario) const
        {
            mpi::communicator world;
            return fmt::format("save_options_{}_{}_np{}", scenario, overload_name(GetParam()), world.size());
        }

        fs::path output_dir() const
        {
            return has_path(GetParam()) ? fs::path("save_options_output") : fs::current_path();
        }

        // The HDF5 file this rank reads back: the shared one for save, its own
        // one for local_save. On a single process, local_save adds no suffix.
        fs::path h5_file(const std::string& name) const
        {
            mpi::communicator world;
            const bool per_rank = is_local(GetParam()) && world.size() > 1;
            const auto stem     = per_rank ? fmt::format("{}_rank{}", name, world.rank()) : name;
            return output_dir() / (stem + ".h5");
        }

        // `flags...` expands into a braced list at each call site, so every
        // call below goes through the same overload resolution as a literal
        // `{true}` or `{true, true}` written by a user.
        template <bool... flags, class Mesh, class Field>
        void save(const std::string& name, const Mesh& mesh, const Field& u) const
        {
            mpi::communicator world;
            switch (GetParam())
            {
                case Overload::path:
                    samurai::save(output_dir(), name, {flags...}, mesh, u);
                    break;
                case Overload::filename:
                    samurai::save(name, {flags...}, mesh, u);
                    break;
                case Overload::path_comm:
                    samurai::save(output_dir(), name, MPI_COMM_WORLD, {flags...}, mesh, u);
                    break;
                case Overload::filename_comm:
                    samurai::save(name, MPI_COMM_WORLD, {flags...}, mesh, u);
                    break;
                case Overload::path_boost_comm:
                    samurai::save(output_dir(), name, world, {flags...}, mesh, u);
                    break;
                case Overload::local_path:
                    samurai::local_save(output_dir(), name, {flags...}, mesh, u);
                    break;
                case Overload::local_filename:
                    samurai::local_save(name, {flags...}, mesh, u);
                    break;
            }
        }

        // Every rank reads the file back once all of them have written it,
        // and checks the groups that tell which options were applied.
        template <class Check>
        void check(const std::string& name, Check&& has_expected_groups) const
        {
            mpi::communicator world;
            world.barrier();
            bool ok = fs::exists(h5_file(name));
            if (ok)
            {
                const HighFive::File file(h5_file(name).string(), HighFive::File::ReadOnly);
                ok = has_expected_groups(file);
            }
            EXPECT_TRUE_ALL_RANKS(ok);
        }

        void remove_files(const std::string& name) const
        {
            mpi::communicator world;
            if (is_local(GetParam()) || world.rank() == 0)
            {
                fs::remove(h5_file(name));
            }
            world.barrier();
        }

        static samurai::Box<double, dim> box()
        {
            return samurai::Box<double, dim>({0., 0.}, {1., 1.});
        }

        static auto make_mr_mesh()
        {
            return samurai::mra::make_mesh(box(), samurai::mesh_config<dim>().min_level(level).max_level(level).disable_args_parse());
        }
    };

    // `{true}` is `by_mesh_id` for a uniform mesh.
    TEST_P(SaveOptions, uniform_mesh_by_mesh_id)
    {
        const samurai::UniformMesh<samurai::UniformConfig<dim>> mesh(box(), level);
        auto u = samurai::make_scalar_field<double>("u", mesh, 1.);

        const auto name = filename("uniform");
        remove_files(name);
        save<true>(name, mesh, u);
        check(name,
              [](const HighFive::File& file)
              {
                  return file.exist("/mesh/cells and ghosts");
              });
    }

    // `{true}` is `by_level` for a multiresolution mesh.
    TEST_P(SaveOptions, mr_mesh_by_level)
    {
        auto mesh = make_mr_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 1.);

        const auto name = filename("mr_level");
        remove_files(name);
        save<true>(name, mesh, u);
        check(name,
              [](const HighFive::File& file)
              {
                  const auto level_mesh = fmt::format("/level/{}/mesh", level);
                  return file.exist(level_mesh) && !file.exist(fmt::format("{}/level_{}_cells", level_mesh, level));
              });
    }

    // `{true, true}` is `by_level` and `by_mesh_id` for a multiresolution mesh.
    TEST_P(SaveOptions, mr_mesh_by_level_and_mesh_id)
    {
        auto mesh = make_mr_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 1.);

        const auto name = filename("mr_level_mesh_id");
        remove_files(name);
        save<true, true>(name, mesh, u);
        check(name,
              [](const HighFive::File& file)
              {
                  return file.exist(fmt::format("/level/{}/mesh/level_{}_cells", level, level));
              });
    }

    INSTANTIATE_TEST_SUITE_P(overloads,
                             SaveOptions,
                             ::testing::Values(Overload::path,
                                               Overload::filename,
                                               Overload::path_comm,
                                               Overload::filename_comm,
                                               Overload::path_boost_comm,
                                               Overload::local_path,
                                               Overload::local_filename),
                             [](const auto& param_info)
                             {
                                 return overload_name(param_info.param);
                             });
}
