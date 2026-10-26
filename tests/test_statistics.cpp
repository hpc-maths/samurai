// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Compiled only when samurai is configured with -DWITH_STATS=ON.

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <samurai/mr/mesh.hpp>
#include <samurai/statistics.hpp>

namespace
{
    namespace fs = std::filesystem;

    /// A stats.json path of its own for the running test, in a fresh directory
    /// removed with everything in it on exit.
    ///
    /// The directory is created under the current working directory rather than
    /// in a shared temporary directory such as /tmp, so no other user can create
    /// or replace it beforehand. Its random suffix keeps concurrent runs of the
    /// test binaries apart, and `create_directory` refuses a name that already
    /// exists, so the test never reuses a directory it did not create.
    class StatsFile
    {
      public:

        StatsFile()
        {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            const auto stem  = std::string("samurai_stats_") + info->test_suite_name() + "_" + info->name() + "_";

            std::random_device random;
            constexpr int max_attempts = 16;
            for (int attempt = 0; attempt < max_attempts && m_dir.empty(); ++attempt)
            {
                auto candidate = fs::current_path() / (stem + std::to_string(random()));
                if (fs::create_directory(candidate))
                {
                    m_dir = std::move(candidate);
                }
            }
            if (m_dir.empty())
            {
                throw std::runtime_error("cannot create a directory for the statistics file under " + fs::current_path().string());
            }
            fs::permissions(m_dir, fs::perms::owner_all, fs::perm_options::replace);
            m_path = m_dir / "stats.json";
        }

        ~StatsFile()
        {
            std::error_code ignored;
            fs::remove_all(m_dir, ignored);
        }

        StatsFile(const StatsFile&)            = delete;
        StatsFile& operator=(const StatsFile&) = delete;
        StatsFile(StatsFile&&)                 = delete;
        StatsFile& operator=(StatsFile&&)      = delete;

        const fs::path& path() const
        {
            return m_path;
        }

        nlohmann::json read() const
        {
            std::ifstream file(m_path);
            return nlohmann::json::parse(file);
        }

      private:

        fs::path m_dir;
        fs::path m_path;
    };

    /// Records the statistics of `mesh` once and returns the entry written for it.
    template <class Mesh>
    nlohmann::json record(const Mesh& mesh)
    {
        StatsFile file;
        {
            samurai::Statistics stats(file.path().string());
            stats("case", mesh);
        }
        return file.read()["case"][0];
    }
}

namespace samurai_test
{
    // Defined in statistics_other_translation_unit.cpp.
    const samurai::Statistics* statistics_in_other_translation_unit();
}

namespace samurai
{
    TEST(statistics, one_recorder_shared_across_translation_units)
    {
        EXPECT_EQ(samurai_test::statistics_in_other_translation_unit(), &statistics);
    }

    TEST(statistics, intervals_per_component_follow_each_axis_in_3d)
    {
        constexpr std::size_t dim   = 3;
        constexpr std::size_t level = 3;
        using Mesh                  = MRMesh<mesh_config<dim>>;

        // The z = 0 plane holds two y-intervals, the z = 2 plane one; every
        // (y, z) row holds a single x-interval.
        typename Mesh::cl_type cl;
        cl[level][{0, 0}].add_point(0);
        cl[level][{2, 0}].add_point(0);
        cl[level][{0, 2}].add_point(0);
        auto mesh = mra::make_mesh(cl, mesh_config<dim>().min_level(level).max_level(level));

        auto stats = record(mesh);

        const auto& by_level = stats["by_level"]["03"];
        EXPECT_EQ(by_level["axis-1"]["number of intervals per component"]["min"], 1);
        EXPECT_EQ(by_level["axis-1"]["number of intervals per component"]["max"], 1);
        EXPECT_EQ(by_level["axis-2"]["number of intervals per component"]["min"], 1);
        EXPECT_EQ(by_level["axis-2"]["number of intervals per component"]["max"], 2);
    }

    TEST(statistics, level_without_cells_between_min_and_max_level)
    {
        constexpr std::size_t dim = 2;
        using Mesh                = MRMesh<mesh_config<dim>>;

        // Real cells at levels 2 and 4 only: level 3 is empty.
        typename Mesh::cl_type cl;
        cl[2][{0}].add_point(0);
        cl[4][{8}].add_point(8);
        auto mesh = mra::make_mesh(cl, mesh_config<dim>().min_level(2).max_level(4));

        auto stats = record(mesh);

        EXPECT_EQ(stats["min_level"], 2);
        EXPECT_EQ(stats["max_level"], 4);
        const auto& empty = stats["by_level"]["03"];
        EXPECT_EQ(empty["cells"], 0);
        for (const auto* axis : {"axis-0", "axis-1"})
        {
            EXPECT_EQ(empty[axis]["number of intervals"], 0) << axis;
            EXPECT_TRUE(empty[axis]["cells per interval"]["min"].is_null()) << axis;
            EXPECT_TRUE(empty[axis]["cells per interval"]["max"].is_null()) << axis;
        }
        EXPECT_TRUE(empty["axis-1"]["number of intervals per component"]["min"].is_null());
        EXPECT_TRUE(empty["axis-1"]["number of intervals per component"]["max"].is_null());

        const auto& finest = stats["by_level"]["04"];
        EXPECT_EQ(finest["cells"], 1);
        EXPECT_EQ(finest["axis-0"]["cells per interval"]["max"], 1);
    }

    TEST(statistics, nothing_written_when_nothing_recorded)
    {
        StatsFile file;
        {
            Statistics stats(file.path().string());
        }
        EXPECT_FALSE(std::filesystem::exists(file.path()));
    }

    TEST(statistics, written_every_save_every_calls_and_on_destruction)
    {
        constexpr std::size_t dim = 1;
        auto mesh                 = mra::make_mesh(Box<double, dim>({0.}, {1.}), mesh_config<dim>().min_level(2).max_level(2));

        StatsFile file;
        {
            Statistics stats(file.path().string(), 2);
            stats("case", mesh);
            EXPECT_FALSE(std::filesystem::exists(file.path()));
            stats("case", mesh);
            EXPECT_EQ(file.read()["case"].size(), 2);
            stats("case", mesh);
            EXPECT_EQ(file.read()["case"].size(), 2);
        }
        EXPECT_EQ(file.read()["case"].size(), 3);
    }
}
