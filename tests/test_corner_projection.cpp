#include <gtest/gtest.h>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    TEST(subset, corner_projection)
    {
        static constexpr std::size_t dim = 2;
        using Config                     = MRConfig<dim>;
        using Box                        = Box<double, dim>;
        using Mesh                       = MRMesh<Config>;
        using interval_t                 = typename Mesh::interval_t;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using direction_t                = DirectionVector<dim>;

        Box box({-1., -1.}, {1., 1.});

        auto mesh_cfg = mesh_config<dim>().min_level(2).max_level(6);

        Mesh mesh{mesh_cfg, box};

        direction_t direction = {-1, -1}; // corner direction
        std::size_t level     = 6;

        auto domain            = self(mesh.domain()).on(level);
        auto fine_inner_corner = difference(
            domain,
            union_(translate(domain, direction_t{-direction[0], 0}), translate(domain, direction_t{0, -direction[1]})));

        bool found     = false;
        std::size_t nb = 0;
        fine_inner_corner(
            [&](const auto& i, const auto& index)
            {
                EXPECT_EQ(i, interval_t(0, 1));
                EXPECT_EQ(index[0], 0);
                ++nb;
                found = true;
            });
        EXPECT_EQ(nb, 1);
        EXPECT_TRUE(found);

        auto fine_outer_corner = intersection(translate(fine_inner_corner, direction), mesh[mesh_id_t::reference][level]);
        found                  = false;
        nb                     = 0;
        fine_outer_corner(
            [&](const auto& i, const auto& index)
            {
                EXPECT_EQ(i, interval_t(-1, 0));
                EXPECT_EQ(index[0], -1);
                ++nb;
                found = true;
            });
        EXPECT_EQ(nb, 1);
        EXPECT_TRUE(found);

        found = false;
        nb    = 0;
        fine_outer_corner.on(level - 1)(
            [&](const auto& i, const auto& index)
            {
                EXPECT_EQ(i, interval_t(-1, 0));
                EXPECT_EQ(index[0], -1);
                ++nb;
                found = true;
            });
        EXPECT_EQ(nb, 1);
        EXPECT_TRUE(found);

        auto parent_ghost = intersection(fine_outer_corner.on(level - 1), mesh[mesh_id_t::reference][level - 1]);
        found             = false;
        nb                = 0;
        parent_ghost(
            [&](const auto& i, const auto& index)
            {
                EXPECT_EQ(i, interval_t(-1, 0));
                EXPECT_EQ(index[0], -1);
                ++nb;
                found = true;
            });
        EXPECT_EQ(nb, 1);
        EXPECT_TRUE(found);
    }
}
