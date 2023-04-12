#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    TEST(bc, scalar_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 1>("u", mesh);

        make_bc<Dirichlet>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 0.);
    }

    TEST(bc, vec_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_field<double, 4>("u", mesh);

        make_bc<Dirichlet>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), xt::zeros<double>({4}));
    }
}
