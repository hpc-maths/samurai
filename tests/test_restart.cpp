#include <gtest/gtest.h>
#include <samurai/amr/mesh.hpp>
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
            EXPECT_TRUE(mesh2.is_box());
        }
        {
            auto mesh = LevelCellArray<2>(4, Box<double, 2>({0, 0}, {1, 1}));
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
            EXPECT_TRUE(mesh2.is_box());
        }
        {
            auto mesh = LevelCellArray<3>(4, Box<double, 3>({0, 0, 0}, {1, 1, 1}));
            decltype(mesh) mesh2;
            dump("mesh", mesh);
            load("mesh", mesh2);
            EXPECT_TRUE(mesh == mesh2);
            EXPECT_TRUE(mesh2.is_box());
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

    TEST(restart, restart_uniform_field)
    {
        using Config = UniformConfig<2>;
        auto mesh    = UniformMesh<Config>(Box<double, 2>({0, 0}, {1, 1}), 3);
        auto u       = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = static_cast<double>(cell.index);
                      });
        auto v = make_vector_field<int, 2>("v", mesh);
        v.fill(2);
        dump("mesh", mesh, u, v);

        decltype(mesh) mesh2;
        auto u2 = make_scalar_field<double>("u", mesh2);
        auto v2 = make_vector_field<int, 2>("v", mesh2);
        load("mesh", mesh2, u2, v2);
        EXPECT_TRUE(mesh == mesh2);
        EXPECT_TRUE(u == u2);
        EXPECT_TRUE(v == v2);
    }

    // Makes the file look as if twice as many processes had written it: every
    // partitioned dataset holds its partitions twice, one after the other.
    static void double_the_writers(HighFive::Group group)
    {
        for (const auto& name : group.listObjectNames())
        {
            if (group.getObjectType(name) != HighFive::ObjectType::Group)
            {
                continue;
            }
            auto child = group.getGroup(name);
            if (!child.exist("partition"))
            {
                double_the_writers(child);
                continue;
            }
            auto bounds   = H5Easy::load<std::vector<std::size_t>>(child.getFile(), child.getPath() + "/partition");
            auto total    = bounds.back();
            auto n_bounds = bounds.size();
            for (std::size_t p = 1; p < n_bounds; ++p)
            {
                bounds.push_back(total + bounds[p]);
            }
            child.unlink("partition");
            child.createDataSet("partition", bounds);

            auto dataset = child.getDataSet("data");
            auto type    = dataset.getDataType();
            std::vector<char> values(2 * total * type.getSize());
            dataset.read_raw(values.data(), type);
            std::copy(values.begin(),
                      values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2),
                      values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2));
            child.unlink("data");
            child.createDataSet("data", HighFive::DataSpace({2 * total}), type).write_raw(values.data(), type);
        }
    }

    // Every process holds the whole uniform mesh, so a file written by any
    // number of processes gives back the same mesh and fields.
    TEST(restart, restart_uniform_other_number_of_processes)
    {
        using Config = UniformConfig<2>;
        auto mesh    = UniformMesh<Config>(Box<double, 2>({0, 0}, {1, 1}), 3);
        auto u       = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = cell.center(0) + 2 * cell.center(1);
                      });
        auto v = make_vector_field<int, 2>("v", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          v[cell][0] = cell.indices[0];
                          v[cell][1] = cell.indices[1];
                      });
        dump("mesh", mesh, u, v);

        // The file is edited without MPI-IO, so a single process must do it.
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        world.barrier();
        if (world.rank() == 0)
#endif
        {
            HighFive::File file("mesh.h5", HighFive::File::ReadWrite);
            const auto n_process = H5Easy::load<std::size_t>(file, "/n_process");
            H5Easy::dump(file, "/n_process", 2 * n_process, H5Easy::DumpMode::Overwrite);
            double_the_writers(file.getGroup("/"));
        }
#ifdef SAMURAI_WITH_MPI
        world.barrier();
#endif

        decltype(mesh) mesh2;
        auto u2 = make_scalar_field<double>("u", mesh2);
        auto v2 = make_vector_field<int, 2>("v", mesh2);
        load("mesh", mesh2, u2, v2);
        EXPECT_TRUE(mesh == mesh2);
        EXPECT_TRUE(u == u2);
        EXPECT_TRUE(v == v2);
    }

    TEST(restart, restart_mrmesh)
    {
        auto mesh = create_mesh<2>(1);
        decltype(mesh) mesh2;
        dump("mesh", mesh);
        load("mesh", mesh2);
        EXPECT_TRUE(mesh == mesh2);
    }

    TEST(restart, restart_box_domain)
    {
        auto mesh = create_mesh<2>(1);
        ASSERT_TRUE(mesh.domain().is_box());
        dump("mesh", mesh);

        decltype(mesh) mesh2;
        load("mesh", mesh2);
        EXPECT_TRUE(mesh2.domain().is_box());
    }

    TEST(restart, restart_non_box_domain)
    {
        CellList<2> cl;
        cl[3][{0}].add_interval({0, 8});
        cl[3][{1}].add_interval({0, 4});
        auto mesh_cfg = mesh_config<2>().min_level(3).max_level(3).disable_minimal_ghost_width();
        auto mesh     = mra::make_mesh(cl, mesh_cfg);
        ASSERT_FALSE(mesh.domain().is_box());
        dump("mesh", mesh);

        decltype(mesh) mesh2;
        load("mesh", mesh2);
        EXPECT_FALSE(mesh2.domain().is_box());
    }

    TEST(restart, restart_file_without_is_box)
    {
        auto mesh = create_mesh<2>(1);
        dump("mesh", mesh);

        // The file is edited without MPI-IO, so a single process must do it:
        // HDF5 locks the file, and every other process would fail to open it.
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        world.barrier();
        if (world.rank() == 0)
#endif
        {
            HighFive::File file("mesh.h5", HighFive::File::ReadWrite);
            file.unlink("/mesh/is_box");
        }
#ifdef SAMURAI_WITH_MPI
        world.barrier();
#endif

        decltype(mesh) mesh2;
        load("mesh", mesh2);
        EXPECT_TRUE(mesh == mesh2);
        EXPECT_FALSE(mesh2.domain().is_box());
    }

    struct mra_factory
    {
        static auto make_mesh(const auto& box, const auto& config)
        {
            return mra::make_mesh(box, config);
        }

        static auto make_empty_mesh(const auto& config)
        {
            return mra::make_empty_mesh(config);
        }
    };

    struct amr_factory
    {
        static auto make_mesh(const auto& box, const auto& config)
        {
            return amr::make_mesh(box, config);
        }

        static auto make_empty_mesh(const auto& config)
        {
            return amr::make_empty_mesh(config);
        }
    };

    template <class Factory>
    class restart_empty_mesh : public ::testing::Test
    {
    };

    using restart_empty_mesh_types = ::testing::Types<mra_factory, amr_factory>;

    TYPED_TEST_SUITE(restart_empty_mesh, restart_empty_mesh_types, );

    // load() rebuilds the mesh from the configuration of the mesh it fills, so an
    // empty mesh must carry the configuration it was created from.
    TYPED_TEST(restart_empty_mesh, keeps_config)
    {
        using Factory = TypeParam;

        auto config = mesh_config<2>().min_level(2).max_level(5).start_level(5).periodic(true).max_stencil_size(6);
        auto mesh   = Factory::make_mesh(Box<double, 2>({0, 0}, {1, 1}), config);
        auto u      = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        dump("mesh_empty", mesh, u);

        auto mesh2 = Factory::make_empty_mesh(config);
        EXPECT_TRUE(mesh2.is_periodic());
        EXPECT_EQ(mesh2.max_stencil_radius(), 3);
        EXPECT_EQ(mesh2.ghost_width(), mesh.ghost_width());

        auto u2 = make_scalar_field<double>("u", mesh2);
        load("mesh_empty", mesh2, u2);

        using mesh_id_t = typename decltype(mesh)::mesh_id_t;
        EXPECT_TRUE(mesh2.is_periodic());
        EXPECT_EQ(mesh2.max_stencil_radius(), 3);
        EXPECT_TRUE(mesh == mesh2);
        EXPECT_TRUE(mesh[mesh_id_t::reference] == mesh2[mesh_id_t::reference]);
        EXPECT_TRUE(u == u2);
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

        dump(
            "mesh",
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
        load(
            "mesh",
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
        dump(
            fs::current_path(),
            "mesh_meta",
            [&](MetadataWriter& w)
            {
                w.time(time);
            },
            mesh);

        decltype(mesh) mesh2;
        double time_read = 0.;
        load(
            fs::current_path(),
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
