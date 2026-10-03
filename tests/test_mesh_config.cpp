#include <set>
#include <stdexcept>

#include <gtest/gtest.h>

#include <samurai/algorithm.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/mesh_config.hpp>

namespace samurai
{
    template <class Mesh>
    auto cell_levels(const Mesh& mesh)
    {
        std::set<std::size_t> levels;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          levels.insert(cell.level);
                      });
        return levels;
    }

    TEST(mesh_config, start_level_defaults_to_max_level)
    {
        auto config = mesh_config<1>().min_level(2).max_level(9);
        EXPECT_EQ(config.start_level(), 9);

        config.max_level(4);
        EXPECT_EQ(config.start_level(), 4);

        config.start_level(3);
        EXPECT_EQ(config.start_level(), 3);
    }

    TEST(mesh_config, amr_mesh_starts_at_max_level_by_default)
    {
        const Box<double, 1> box({0}, {1});

        auto config = mesh_config<1>().min_level(8).max_level(8);
        auto mesh   = amr::make_mesh(box, config);
        EXPECT_EQ(cell_levels(mesh), std::set<std::size_t>{8});
        EXPECT_EQ(mesh.nb_cells(amr::AMR_Id::cells), 256);

        // The constructor does not go through parse_args
        amr::Mesh<mesh_config<1>> direct_mesh(box, mesh_config<1>().min_level(2).max_level(4));
        EXPECT_EQ(cell_levels(direct_mesh), std::set<std::size_t>{4});
    }

    TEST(mesh_config, amr_mesh_starts_at_the_start_level)
    {
        const Box<double, 1> box({0}, {1});

        auto config = mesh_config<1>().min_level(2).max_level(8).start_level(5);
        auto mesh   = amr::make_mesh(box, config);
        EXPECT_EQ(cell_levels(mesh), std::set<std::size_t>{5});
    }

    TEST(mesh_config, start_level_outside_the_level_range_throws)
    {
        const Box<double, 1> box({0}, {1});

        EXPECT_THROW(amr::make_mesh(box, mesh_config<1>().min_level(2).max_level(5).start_level(7)), std::invalid_argument);
        EXPECT_THROW(amr::make_mesh(box, mesh_config<1>().min_level(4).max_level(5).start_level(3)), std::invalid_argument);

        EXPECT_THROW((amr::Mesh<mesh_config<1>>(box, mesh_config<1>().min_level(2).max_level(5).start_level(7))), std::invalid_argument);
    }
}
