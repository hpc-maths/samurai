#include <gtest/gtest.h>
#include <samurai/amr/mesh.hpp>

namespace samurai
{
    auto create_meshes(std::size_t level)
    {
        using Config  = mesh_config<1>;
        using Mesh    = decltype(amr::make_empty_Mesh(std::declval<Config>()));
        using cl_type = typename Mesh::cl_type;

        cl_type cl1;
        cl1[level][{}].add_interval({-1, 2});
        cl_type cl2;
        cl2[level][{}].add_interval({0, 3});

        auto mesh_cfg = mesh_config<1>().min_level(level).max_level(level).disable_minimal_ghost_width();
        auto m1       = amr::make_Mesh(mesh_cfg, cl1);
        auto m2       = amr::make_Mesh(mesh_cfg, cl2);
        return std::make_tuple(m1, m2);
    }

    TEST(set, for_each_interval)
    {
        using Config    = mesh_config<1>;
        using Mesh      = decltype(amr::make_empty_Mesh(std::declval<Config>()));
        using mesh_id_t = typename Mesh::mesh_id_t;

        std::size_t level = 1;
        auto meshes       = create_meshes(level);
        auto& m1          = std::get<0>(meshes);
        auto& m2          = std::get<1>(meshes);
        auto set          = intersection(m1[mesh_id_t::cells][level], m2[mesh_id_t::cells][level]);

        int nb_intervals = 0;
        for_each_interval(set,
                          [&](auto l, auto& i, auto&)
                          {
                              nb_intervals++;
                              EXPECT_EQ(l, level);
                              EXPECT_EQ(i.start, 0);
                              EXPECT_EQ(i.end, 2);
                          });
        EXPECT_EQ(nb_intervals, 1);
    }

    TEST(set, for_each_cell)
    {
        using Config = mesh_config<1>;

        std::size_t level = 1;
        auto meshes       = create_meshes(level);
        auto& m1          = std::get<0>(meshes);
        auto& m2          = std::get<1>(meshes);

        using Mesh      = std::tuple_element_t<0, decltype(meshes)>;
        using mesh_id_t = Mesh::mesh_id_t;

        auto set = intersection(m1[mesh_id_t::cells][level], m2[mesh_id_t::cells][level]);

        /** Cell indices in m1:
         *
         *  -2  -1   0   1   2   3
         *   |...|---|---|---|...|
         *     0   1   2   3   4
         *
         * set = [0,2[
         */

        int nb_cells = 0;
        for_each_cell(m1,
                      set,
                      [&](auto& cell)
                      {
                          if (nb_cells == 0)
                          {
                              EXPECT_EQ(cell.center(0), 0.25);
                              EXPECT_EQ(cell.index, 2);
                          }
                          else if (nb_cells == 1)
                          {
                              EXPECT_EQ(cell.center(0), 0.75);
                              EXPECT_EQ(cell.index, 3);
                          }
                          nb_cells++;
                      });
        EXPECT_EQ(nb_cells, 2);
    }
}
