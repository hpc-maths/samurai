#include <gtest/gtest.h>
#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    template <class Mesh>
    auto init(Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        auto u          = make_field<double, 1>("u", mesh);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          u[cell] = xt::sum(cell.center())[0];
                      });
        return u;
    }

    TEST(portion, 1D)
    {
        constexpr std::size_t dim = 1;
        using config              = UniformConfig<dim>;
        using interval_t          = typename UniformConfig<dim>::interval_t;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0}, {1}), 5);
        auto u                    = init(mesh);

        auto p = portion<1>(u, 5, interval_t{2, 3}, 4, 0);
        EXPECT_EQ(p[0], ((2 << 4) + .5) / (1 << 9));
    }

    TEST(portion, 2D)
    {
        constexpr std::size_t dim = 2;
        using config              = UniformConfig<dim>;
        using interval_t          = typename UniformConfig<dim>::interval_t;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0}, {1, 1}), 5);
        auto u                    = init(mesh);

        auto p = portion<1>(u, 5, interval_t{2, 3}, 2, 4, 0, 0);
        EXPECT_EQ(p[0], 2 * ((2 << 4) + .5) / (1 << 9));
    }

    TEST(portion, 3D)
    {
        constexpr std::size_t dim = 3;
        using config              = UniformConfig<dim>;
        using interval_t          = typename UniformConfig<dim>::interval_t;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0, 0}, {1, 1, 1}), 5);
        auto u                    = init(mesh);

        auto p = portion<1>(u, 5, interval_t{2, 3}, 2, 2, 4, 0, 0, 0);
        EXPECT_EQ(p[0], 3 * ((2 << 4) + .5) / (1 << 9));
    }
}
