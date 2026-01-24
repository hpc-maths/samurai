#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    TEST(field, from_expr)
    {
        Box<double, 1> box{{0}, {1}};
        // auto mesh    = MRMesh<Config>(box, 3, 3);

        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 3);

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        using field_t = decltype(u);
        field_t ue    = 5 + u;

        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          EXPECT_EQ(ue[cell], 6);
                      });
    }

    TEST(field, copy_from_const)
    {
        Box<double, 1> box{{0}, {1}};
        using Config       = UniformConfig<1>;
        auto mesh          = UniformMesh<Config>(box, 3);
        const auto u_const = make_scalar_field<double>("uc", mesh, 1.);

        auto u = u_const;
        EXPECT_EQ(u.name(), u_const.name());
        EXPECT_TRUE(compare(u.array(), u_const.array()));
        EXPECT_EQ(u.mesh(), u_const.mesh());
        EXPECT_EQ(&(u.mesh()), &(u_const.mesh()));

        auto m              = holder(mesh);
        const auto u_const1 = make_scalar_field<double>("uc", m, 1.);
        auto u1             = u_const1;
        EXPECT_EQ(u1.name(), u_const1.name());
        EXPECT_TRUE(compare(u1.array(), u_const1.array()));
        EXPECT_EQ(u1.mesh(), u_const1.mesh());
    }

    TEST(field, copy_assignment)
    {
        Box<double, 1> box{{0}, {1}};
        using Config       = UniformConfig<1>;
        auto mesh1         = UniformMesh<Config>(box, 5);
        auto mesh2         = UniformMesh<Config>(box, 3);
        const auto u_const = make_scalar_field<double>("uc",
                                                       mesh1,
                                                       [](const auto& coords)
                                                       {
                                                           return coords[0];
                                                       });
        auto u             = make_scalar_field<double>("u",
                                           mesh2,
                                           [](const auto& coords)
                                           {
                                               return coords[0];
                                           });

        u = u_const;
        EXPECT_EQ(u.name(), u_const.name());
        EXPECT_TRUE(compare(u.array(), u_const.array()));
        EXPECT_EQ(u.mesh(), u_const.mesh());
        EXPECT_EQ(&(u.mesh()), &(u_const.mesh()));

        auto m1             = holder(mesh1);
        auto m2             = holder(mesh2);
        const auto u_const1 = make_scalar_field<double>("uc",
                                                        m1,
                                                        [](const auto& coords)
                                                        {
                                                            return coords[0];
                                                        });
        auto u1             = make_scalar_field<double>("u",
                                            m2,
                                            [](const auto& coords)
                                            {
                                                return coords[0];
                                            });
        u1                  = u_const1;
        EXPECT_EQ(u1.name(), u_const1.name());
        EXPECT_TRUE(compare(u1.array(), u_const1.array()));
        EXPECT_EQ(u1.mesh(), u_const1.mesh());
    }

    TEST(field, iterator)
    {
        CellList<2> cl;
        cl[1][{0}].add_interval({0, 2});
        cl[1][{0}].add_interval({4, 6});
        cl[2][{0}].add_interval({4, 8});

        auto mesh_cfg = mesh_config<2>().min_level(1).max_level(2);
        auto mesh     = mra::make_mesh(cl, mesh_cfg);
        auto field    = make_scalar_field<std::size_t>("u", mesh);

        std::size_t index = 0;
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          field[cell] = index++;
                      });

        auto it = field.begin();
        EXPECT_TRUE(compare(*it, samurai::Array<std::size_t, 2, true>{0, 1}));
        it += 2;
        EXPECT_TRUE(compare(*it, samurai::Array<std::size_t, 4, true>{4, 5, 6, 7}));
        ++it;
        EXPECT_EQ(it, field.end());

        auto itr = field.rbegin();
        EXPECT_TRUE(compare(*itr, samurai::Array<std::size_t, 4, true>{4, 5, 6, 7}));
        itr += 2;
        EXPECT_TRUE(compare(*itr, samurai::Array<std::size_t, 2, true>{0, 1}));
        ++itr;
        EXPECT_EQ(itr, field.rend());
    }

    TEST(field, name)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 5);
        auto u       = make_scalar_field<double>("u", mesh);

        EXPECT_EQ(u.name(), "u");
        u.name() = "new_name";
        EXPECT_EQ(u.name(), "new_name");
    }

    TEST(field, equal_operator_scalar)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 5);
        auto u1      = make_scalar_field<double>("u", mesh, 1.0);
        auto u2      = make_scalar_field<double>("u", mesh, 1.0);
        auto u3      = make_scalar_field<double>("u", mesh, 2.0);

        EXPECT_TRUE(u1 == u2);
        EXPECT_FALSE(u1 != u2);
        EXPECT_FALSE(u1 == u3);
        EXPECT_TRUE(u1 != u3);

        auto v1 = make_scalar_field<int>("v", mesh, 1);
        auto v2 = make_scalar_field<int>("v", mesh, 1);
        auto v3 = make_scalar_field<int>("v", mesh, 2);

        EXPECT_TRUE(v1 == v2);
        EXPECT_FALSE(v1 != v2);
        EXPECT_FALSE(v1 == v3);
        EXPECT_TRUE(v1 != v3);
    }

    TEST(field, equal_operator_vector)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 5);
        auto u1      = make_vector_field<double, 3>("u", mesh, 1.0);
        auto u2      = make_vector_field<double, 3>("u", mesh, 1.0);
        auto u3      = make_vector_field<double, 3>("u", mesh, 2.0);

        EXPECT_TRUE(u1 == u2);
        EXPECT_FALSE(u1 != u2);
        EXPECT_FALSE(u1 == u3);
        EXPECT_TRUE(u1 != u3);

        auto v1 = make_vector_field<int, 3>("v", mesh, 1);
        auto v2 = make_vector_field<int, 3>("v", mesh, 1);
        auto v3 = make_vector_field<int, 3>("v", mesh, 2);

        EXPECT_TRUE(v1 == v2);
        EXPECT_FALSE(v1 != v2);
        EXPECT_FALSE(v1 == v3);
        EXPECT_TRUE(v1 != v3);
    }

}
