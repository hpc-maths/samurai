#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

#include <xtensor/xtensor.hpp>

namespace samurai
{
    TEST(bc, scalar_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 1>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 0.);
    }

    TEST(bc, vec_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), zeros<double>(4)));
    }

    TEST(bc, scalar_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 1>("u", mesh);

        make_bc<Dirichlet<1>>(u, 2);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 2);
    }

    TEST(bc, vec_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u, 1., 2., 3., 4.);
        samurai::Array<double, 4, false> expected({1, 2, 3, 4});
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), expected));
    }

    TEST(bc, scalar_function)
    {
        static constexpr std::size_t dim = 1;
        using config                     = MRConfig<dim>;

        Box<double, dim> box = {{0}, {1}};
        auto mesh            = MRMesh<config>(box, 2, 4);
        auto u               = make_field<double, 1>("u", mesh);

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

}
