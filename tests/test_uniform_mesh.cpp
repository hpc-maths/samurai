#include <gtest/gtest.h>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    TEST(uniform_mesh, get_index_1d)
    {
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(Box<double, 1>({0}, {1}), 3);

        std::size_t n_cells = 0;
        for_each_interval(mesh[UniformMeshId::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  EXPECT_EQ(mesh.get_index(level, i, index), mesh.get_index(i));
                                  ++n_cells;
                              }
                          });
        EXPECT_EQ(n_cells, 8);
    }

    TEST(uniform_mesh, get_index_2d)
    {
        using Config = UniformConfig<2>;
        auto mesh    = UniformMesh<Config>(Box<double, 2>({0, 0}, {1, 1}), 3);

        std::size_t n_cells = 0;
        for_each_interval(mesh[UniformMeshId::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  EXPECT_EQ(mesh.get_index(level, i, index), mesh.get_index(i, index[0]));
                                  ++n_cells;
                              }
                          });
        EXPECT_EQ(n_cells, 64);
    }
}
