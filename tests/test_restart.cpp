#include <gtest/gtest.h>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    template <std::size_t dim>
    auto create_mesh(double box_boundary)
    {
        using Box     = samurai::Box<double, dim>;
        using point_t = typename Box::point_t;

        point_t box_corner1, box_corner2;
        box_corner1.fill(0);
        box_corner2.fill(box_boundary);
        Box box(box_corner1, box_corner2);
        auto mesh_cfg = samurai::mesh_config<dim>().min_level(2).max_level(5).disable_minimal_ghost_width();

        return mra::make_mesh(box, mesh_cfg);
    }

    TEST(restart, restart_lca)
    {
        {
            auto mesh = LevelCellArray<1>(4, Box<double, 1>(0, 1));
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
        {
            auto mesh = LevelCellArray<2>(4, Box<double, 2>({0, 0}, {1, 1}));
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
        {
            auto mesh = LevelCellArray<3>(4, Box<double, 3>({0, 0, 0}, {1, 1, 1}));
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
    }

    TEST(restart, restart_ca)
    {
        {
            CellList<1> cl;
            cl[1][{}].add_interval({2, 8});
            cl[1][{}].add_interval({10, 15});
            cl[4][{}].add_interval({0, 1});
            auto mesh = CellArray<1>(cl);
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
        {
            CellList<2> cl;
            cl[1][{0}].add_interval({2, 8});
            cl[1][{1}].add_interval({10, 15});
            cl[4][{6}].add_interval({0, 1});
            auto mesh = CellArray<2>(cl);
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
        {
            CellList<3> cl;
            cl[1][{0, 0}].add_interval({2, 8});
            cl[1][{1, 0}].add_interval({10, 15});
            cl[4][{6, 1}].add_interval({0, 1});
            auto mesh = CellArray<3>(cl);
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
        }
    }

    TEST(restart, restart_uniform)
    {
        using Config = UniformConfig<2>;
        auto mesh    = UniformMesh<Config>(Box<double, 2>({0, 0}, {1, 1}), 3);
        decltype(mesh) mesh2;
        dump("mesh", mesh);
        load("mesh", mesh2);
        EXPECT_TRUE(mesh == mesh2);
    }

    TEST(restart, restart_mrmesh)
    {
        auto mesh = create_mesh<2>(1);
        decltype(mesh) mesh2;
        dump("mesh", mesh);
        load("mesh", mesh2);
        EXPECT_TRUE(mesh == mesh2);
    }

    TEST(restart, restart_field)
    {
        auto mesh = create_mesh<2>(1);
        auto u    = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        dump("mesh", mesh, u);

        auto mesh2 = create_mesh<2>(10);
        auto u2    = make_scalar_field<double>("u", mesh2);
        load("mesh", mesh2, u2);
        EXPECT_TRUE(u == u2);
    }

    TEST(restart, restart_multiple_fields)
    {
        auto mesh = create_mesh<2>(1);
        auto u    = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        auto v = make_vector_field<int, 2>("v", mesh);
        v.fill(2);
        dump("mesh", mesh, u, v);
        auto mesh2 = create_mesh<2>(10);
        auto u2    = make_scalar_field<double>("u", mesh2);
        auto v2    = make_vector_field<int, 2>("v", mesh2);
        load("mesh", mesh2, u2, v2);
        EXPECT_TRUE(u == u2);
        EXPECT_TRUE(v == v2);
    }

    TEST(restart, restart_metadata)
    {
        auto mesh = create_mesh<2>(1);
        auto u    = make_scalar_field<double>("u", mesh);
        u.fill(1.);

        const double time = 1.5;
        const double Re   = 100.;
        const int iter    = 42;

        dump("mesh",
             [&](MetadataWriter& w)
             {
                 w.time(time).attribute("Re", Re).attribute("iteration", iter);
             },
             mesh,
             u);

        auto mesh2 = create_mesh<2>(10);
        auto u2    = make_scalar_field<double>("u", mesh2);

        double time_read = 0.;
        double Re_read   = 0.;
        int iter_read    = 0;
        bool has_time    = false;
        bool has_missing = true;
        load("mesh",
             [&](MetadataReader& r)
             {
                 has_time    = r.has("time");
                 has_missing = r.has("does_not_exist");
                 time_read   = r.time();
                 Re_read     = r.attribute<double>("Re");
                 iter_read   = r.attribute<int>("iteration");
             },
             mesh2,
             u2);

        EXPECT_TRUE(u == u2); // fields still round-trip alongside the metadata
        EXPECT_TRUE(has_time);
        EXPECT_FALSE(has_missing);
        EXPECT_DOUBLE_EQ(time_read, time);
        EXPECT_DOUBLE_EQ(Re_read, Re);
        EXPECT_EQ(iter_read, iter);
    }

    TEST(restart, restart_metadata_with_path)
    {
        auto mesh         = create_mesh<2>(1);
        const double time = 2.5;
        dump(fs::current_path(),
             "mesh_meta",
             [&](MetadataWriter& w)
             {
                 w.time(time);
             },
             mesh);

        decltype(mesh) mesh2;
        double time_read = 0.;
        load(fs::current_path(),
             "mesh_meta",
             [&](MetadataReader& r)
             {
                 time_read = r.time();
             },
             mesh2);

        EXPECT_TRUE(mesh == mesh2);
        EXPECT_DOUBLE_EQ(time_read, time);
    }
}
