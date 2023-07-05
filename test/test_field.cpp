#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{

    TEST(field, copy_from_const)
    {
        samurai::Box<double, 1> box{{0}, {1}};
        using Config       = samurai::UniformConfig<1>;
        auto mesh          = samurai::UniformMesh<Config>(box, 3);
        const auto u_const = samurai::make_field<double, 1>("uc", mesh);

        auto u = u_const;

        auto m              = samurai::holder(mesh);
        const auto u_const1 = samurai::make_field<double, 1>("uc", m);
        auto u1             = u_const1;
    }

    TEST(field, iterator)
    {
        using config = MRConfig<2>;
        CellList<2> cl;
        cl[1][{0}].add_interval({0, 2});
        cl[1][{0}].add_interval({4, 6});
        cl[2][{0}].add_interval({4, 8});

        auto mesh  = MRMesh<config>(cl, 1, 2);
        auto field = make_field<std::size_t, 1>("u", mesh);

        std::size_t index = 0;
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          field[cell] = index++;
                      });

        auto it = field.begin();
        EXPECT_EQ(*it, (xt::xtensor<std::size_t, 1>{0, 1}));
        it += 2;
        EXPECT_EQ(*it, (xt::xtensor<std::size_t, 1>{4, 5, 6, 7}));
        ++it;
        EXPECT_EQ(it, field.end());

        auto itr = field.rbegin();
        EXPECT_EQ(*itr, (xt::xtensor<std::size_t, 1>{4, 5, 6, 7}));
        itr += 2;
        EXPECT_EQ(*itr, (xt::xtensor<std::size_t, 1>{0, 1}));
        ++itr;
        EXPECT_EQ(itr, field.rend());
    }
}
