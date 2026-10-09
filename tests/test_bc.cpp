#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

#include <xtensor/containers/xtensor.hpp>

namespace samurai
{
    TEST(bc, scalar_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 0.);
    }

    TEST(bc, vec_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_vector_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), zeros<double>(4)));
    }

    TEST(bc, scalar_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u, 2);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 2);
    }

    TEST(bc, vec_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_vector_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u, 1., 2., 3., 4.);
        samurai::Array<double, 4> expected({1, 2, 3, 4});
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), expected));
    }

    TEST(bc, scalar_function)
    {
        static constexpr std::size_t dim = 1;

        Box<double, dim> box = {{0}, {1}};
        auto mesh_cfg        = mesh_config<dim>().min_level(2).max_level(4);
        auto mesh            = mra::make_mesh(box, mesh_cfg);
        auto u               = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  return 0;
                              });

        using cell_t   = typename decltype(u)::cell_t;
        using coords_t = typename cell_t::coords_t;
        cell_t cell;
        coords_t coords = {0.};
        EXPECT_EQ(u.get_bc()[0]->value({1}, cell, coords), 0);
    }

    // Regression test for #577: Bc::on must accept a predicate on the coordinates of the boundary faces.
    TEST(bc, on_predicate)
    {
        static constexpr std::size_t dim = 2;

        Box<double, dim> box({0., 0.}, {1., 1.});
        auto mesh_cfg = mesh_config<dim>().min_level(3).max_level(3);
        auto mesh     = mra::make_mesh(box, mesh_cfg);
        auto u        = make_scalar_field<double>("u", mesh);
        auto v        = make_scalar_field<double>("v", mesh);

        auto pred = [](const auto& x)
        {
            return x[0] < 0.5;
        };
        const auto& region     = make_bc<Dirichlet<1>>(u, 1.)->on(pred)->get_region();
        const auto& ref_region = make_bc<Dirichlet<1>>(v, 1.)->on(make_bc_region(mesh, pred))->get_region();

        // 8 cells on the left side, 4 on the bottom side and 4 on the top side, none on the right side.
        ASSERT_EQ(region.first.size(), 3u);
        std::size_t nb_cells = 0;
        for (const auto& lca : region.second)
        {
            nb_cells += lca.nb_cells();
        }
        EXPECT_EQ(nb_cells, 16u);

        ASSERT_EQ(region.first.size(), ref_region.first.size());
        for (std::size_t i = 0; i < region.first.size(); ++i)
        {
            EXPECT_EQ(region.first[i], ref_region.first[i]);
            EXPECT_EQ(region.second[i], ref_region.second[i]);
        }
    }
}
