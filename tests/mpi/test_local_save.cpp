// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// local_save is not collective: each rank that calls it writes
// <filename>_rank<rank>.h5 and .xdmf, holding only its own cells. Every
// overload is checked, with all ranks calling it and with only some of them.

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace fs  = std::filesystem;
namespace mpi = boost::mpi;

namespace
{
    constexpr std::size_t dim = 2;

    enum class Overload
    {
        path_options,
        path,
        filename_options,
        filename
    };

    std::string overload_name(Overload overload)
    {
        switch (overload)
        {
            case Overload::path_options:
                return "path_options";
            case Overload::path:
                return "path";
            case Overload::filename_options:
                return "filename_options";
            case Overload::filename:
                return "filename";
        }
        return "unknown";
    }

    class LocalSave : public samurai_test::MpiTest,
                      public ::testing::WithParamInterface<Overload>
    {
      protected:

        // The tests for 2, 3 and 4 processes may run at the same time in the
        // same working directory, so the file names carry the process count.
        std::string filename(const std::string& scenario) const
        {
            mpi::communicator world;
            return fmt::format("local_save_{}_{}_np{}", scenario, overload_name(GetParam()), world.size());
        }

        // Directory the overload under test writes into: the overloads
        // without one write into the current directory.
        fs::path output_dir() const
        {
            const bool with_path = GetParam() == Overload::path_options || GetParam() == Overload::path;
            return with_path ? fs::path("local_save_output") : fs::current_path();
        }

        template <class Mesh, class Field>
        void local_save(const std::string& name, const Mesh& mesh, const Field& u) const
        {
            switch (GetParam())
            {
                case Overload::path_options:
                    samurai::local_save(output_dir(), name, samurai::Hdf5Options<Mesh>{}, mesh, u);
                    break;
                case Overload::path:
                    samurai::local_save(output_dir(), name, mesh, u);
                    break;
                case Overload::filename_options:
                    samurai::local_save(name, samurai::Hdf5Options<Mesh>{}, mesh, u);
                    break;
                case Overload::filename:
                    samurai::local_save(name, mesh, u);
                    break;
            }
        }

        // Writes a field holding the rank number from the ranks selected by
        // `writes`, then checks that each of them got its own file pair, with
        // its own cells only.
        template <class Predicate>
        void check(const std::string& scenario, Predicate writes) const
        {
            mpi::communicator world;
            const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
            auto mesh = samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(4).max_level(4).disable_args_parse());
            auto u    = samurai::make_scalar_field<double>("u", mesh, static_cast<double>(world.rank()));

            const std::string name = filename(scenario);
            const auto local_name  = fmt::format("{}_rank{}", name, world.rank());

            const auto dir = output_dir();
            // Files left by an earlier run must not let the test pass.
            fs::remove(dir / (local_name + ".xdmf"));
            fs::remove(dir / (local_name + ".h5"));

            bool ok = true;
            if (writes(world.rank()))
            {
                local_save(name, mesh, u);

                ok = ok && fs::exists(dir / (local_name + ".xdmf"));
                ok = ok && fs::exists(dir / (local_name + ".h5"));
                if (ok)
                {
                    HighFive::File file((dir / (local_name + ".h5")).string(), HighFive::File::ReadOnly);
                    // A file written by one process has no /mesh/rank_<rank> groups.
                    for (const auto& object : file.getGroup("/mesh").listObjectNames())
                    {
                        ok = ok && !object.starts_with("rank_");
                    }

                    // Checked before loading: an exception on one rank would
                    // leave the others waiting in EXPECT_TRUE_ALL_RANKS.
                    ok = ok && file.exist("/mesh/fields/u");
                    if (ok)
                    {
                        auto values = H5Easy::load<std::vector<double>>(file, "/mesh/fields/u");
                        ok          = values.size() == mesh.nb_cells(decltype(mesh)::mesh_id_t::cells);
                        for (double v : values)
                        {
                            ok = ok && v == static_cast<double>(world.rank());
                        }
                    }
                }
            }
            EXPECT_TRUE_ALL_RANKS(ok);
        }
    };

    TEST_P(LocalSave, every_rank_writes_its_own_cells)
    {
        check("all",
              [](int)
              {
                  return true;
              });
    }

    // A collective save would block rank 0 until the other ranks join it.
    TEST_P(LocalSave, only_some_ranks_write)
    {
        check("some",
              [](int rank)
              {
                  return rank % 2 == 0;
              });
    }

    INSTANTIATE_TEST_SUITE_P(overloads,
                             LocalSave,
                             ::testing::Values(Overload::path_options, Overload::path, Overload::filename_options, Overload::filename),
                             [](const auto& param_info)
                             {
                                 return overload_name(param_info.param);
                             });
}
