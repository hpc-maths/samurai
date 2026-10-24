#include <map>

#include <gtest/gtest.h>

#ifdef SAMURAI_WITH_MPI
#include <mpi.h>
#endif

#include <samurai/arguments.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    template <typename T>
    class hdf5_test : public ::testing::Test
    {
    };

    using hdf5_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(hdf5_test, hdf5_test_types, );

    template <typename mesh_t>
    void test_save(const mesh_t& mesh)
    {
        save("test_save_mesh", mesh);
        save("test", "test_save_mesh", mesh);
        save("test", "test_save_mesh", {true, true}, mesh);
        save("test_save_mesh", Hdf5Options<mesh_t>{true, true}, mesh);
        local_save("test_save_mesh", mesh);
        local_save("test", "test_save_mesh", mesh);
        local_save("test", "test_save_mesh", {true, true}, mesh);
        local_save("test_save_mesh", Hdf5Options<mesh_t>{true, true}, mesh);
#ifdef SAMURAI_WITH_MPI
        save("test_save_mesh", MPI_COMM_WORLD, mesh);
        save("test", "test_save_mesh", MPI_COMM_WORLD, mesh);
        save("test", "test_save_mesh", MPI_COMM_WORLD, {true, true}, mesh);
        save("test_save_mesh", MPI_COMM_WORLD, {true, true}, mesh);

        save("test_save_mesh", MPI_COMM_SELF, mesh);
        save("test", "test_save_mesh", MPI_COMM_SELF, mesh);
        save("test", "test_save_mesh", MPI_COMM_SELF, {true, true}, mesh);
        save("test_save_mesh", MPI_COMM_SELF, {true, true}, mesh);
#endif
    }

    template <typename config_t>
    void test_save_uniform(const UniformMesh<config_t>& mesh)
    {
        save("test_save_mesh", mesh);
        save("test", "test_save_mesh", mesh);
        save("test", "test_save_mesh", Hdf5Options<UniformMesh<config_t>>{true}, mesh);
        save("test_save_mesh", Hdf5Options<UniformMesh<config_t>>{true}, mesh);
        local_save("test_save_mesh", mesh);
        local_save("test", "test_save_mesh", mesh);
        local_save("test", "test_save_mesh", Hdf5Options<UniformMesh<config_t>>{true}, mesh);
        local_save("test_save_mesh", Hdf5Options<UniformMesh<config_t>>{true}, mesh);
#ifdef SAMURAI_WITH_MPI
        save("test_save_mesh", MPI_COMM_WORLD, mesh);
        save("test", "test_save_mesh", MPI_COMM_WORLD, mesh);
        save("test", "test_save_mesh", MPI_COMM_WORLD, {true}, mesh);
        save("test_save_mesh", MPI_COMM_WORLD, {true}, mesh);

        save("test_save_mesh", MPI_COMM_SELF, mesh);
        save("test", "test_save_mesh", MPI_COMM_SELF, mesh);
        save("test", "test_save_mesh", MPI_COMM_SELF, {true}, mesh);
        save("test_save_mesh", MPI_COMM_SELF, {true}, mesh);
#endif
    }

    TYPED_TEST(hdf5_test, cell_array)
    {
        static constexpr std::size_t dim = TypeParam::value;
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        CellArray<dim> ca;
        ca[4] = {4, box};
        test_save(ca);
        args::save_debug_fields = true;
        test_save(ca);
    }

    TYPED_TEST(hdf5_test, level_cell_array)
    {
        static constexpr std::size_t dim = TypeParam::value;
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        LevelCellArray<dim> lca(4, box);
        test_save(lca);
        args::save_debug_fields = true;
        test_save(lca);
    }

    TYPED_TEST(hdf5_test, uniform_mesh)
    {
        static constexpr std::size_t dim = TypeParam::value;
        using Config                     = UniformConfig<dim>;
        using Mesh                       = UniformMesh<Config>;
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        Mesh uniform(box, 4);
        test_save_uniform(uniform);
        args::save_debug_fields = true;
        test_save_uniform(uniform);
    }

    TYPED_TEST(hdf5_test, mr_mesh)
    {
        static constexpr std::size_t dim = TypeParam::value;

        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        auto mesh_cfg = mesh_config<dim>().min_level(4).max_level(4);
        auto mesh     = mra::make_mesh(box, mesh_cfg);
        test_save(mesh);
        args::save_debug_fields = true;
        test_save(mesh);
    }

    TEST(hdf5, vertices_shared_with_non_dyadic_cell_length)
    {
        // With a scaling factor of 0.1 the cell lengths are not dyadic numbers, so the physical
        // coordinates of a vertex shared by several cells, or by two levels, are not bitwise
        // equal when computed from each cell. The vertex must still be written only once.
        static constexpr std::size_t dim = 2;
        CellList<dim> cl({-1., -1.}, 0.1);
        for (int j = 0; j < 40; ++j)
        {
            cl[3][{j}].add_interval({0, 20});
        }
        for (int j = 0; j < 80; ++j)
        {
            cl[4][{j}].add_interval({40, 80});
        }
        CellArray<dim> ca(cl);

        auto [coords, connectivity] = extract_coords_and_connectivity(ca);

        // 21 x 41 vertices on level 3 and 41 x 81 on level 4, the 41 level 3 vertices of the
        // interface being level 4 vertices too.
        constexpr std::size_t expected_nb_points = 21 * 41 + 41 * 81 - 41;
        ASSERT_EQ(coords.shape(0), expected_nb_points);
        ASSERT_EQ(connectivity.shape(0), ca.nb_cells());

        // Every cell must reference vertices at its own corners.
        std::size_t index = 0;
        for_each_cell(ca,
                      [&](const auto& cell)
                      {
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              EXPECT_NEAR(coords(connectivity(index, 0), d), cell.corner(d), 1e-14);
                              EXPECT_NEAR(coords(connectivity(index, 2), d), cell.corner(d) + cell.length, 1e-14);
                          }
                          ++index;
                      });
    }

    TYPED_TEST(hdf5_test, metadata)
    {
        static constexpr std::size_t dim = TypeParam::value;

        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        auto mesh_cfg = mesh_config<dim>().min_level(4).max_level(4);
        auto mesh     = mra::make_mesh(box, mesh_cfg);
        auto u        = make_scalar_field<double>("u", mesh);
        u.fill(1.);

        const double time = 1.5;
        const double Re   = 100.;

        Hdf5Options<decltype(mesh)> options;
        options.metadata = [&](MetadataWriter& w)
        {
            w.time(time).attribute("Re", Re).information("scheme", "WENO5");
        };
        save("test", "test_metadata", options, mesh, u);

        // The metadata callback is optional: saving without it must still work
        // and produce no <Time> node.
        save("test", "test_no_metadata", mesh, u);

#ifndef SAMURAI_WITH_MPI
        // --- HDF5 attributes ---
        {
            HighFive::File file("test/test_metadata.h5", HighFive::File::ReadOnly);
            ASSERT_TRUE(file.hasAttribute("time"));
            ASSERT_TRUE(file.hasAttribute("Re"));
            EXPECT_DOUBLE_EQ(file.getAttribute("time").read<double>(), time);
            EXPECT_DOUBLE_EQ(file.getAttribute("Re").read<double>(), Re);
        }

        // --- XDMF <Time> in every top-level grid + <Information> ---
        {
            pugi::xml_document doc;
            ASSERT_TRUE(doc.load_file("test/test_metadata.xdmf"));
            auto domain        = doc.child("Xdmf").child("Domain");
            std::size_t n_grid = 0;
            std::size_t n_time = 0;
            for (pugi::xml_node grid : domain.children("Grid"))
            {
                ++n_grid;
                auto time_node = grid.child("Time");
                if (time_node)
                {
                    ++n_time;
                    EXPECT_DOUBLE_EQ(time_node.attribute("Value").as_double(), time);
                }
            }
            EXPECT_GT(n_grid, 0u);
            EXPECT_EQ(n_grid, n_time); // a <Time> node in every top-level grid
            EXPECT_STREQ(domain.child("Information").attribute("Name").value(), "scheme");
            EXPECT_STREQ(domain.child("Information").attribute("Value").value(), "WENO5");
        }

        // --- no metadata -> no attribute, no <Time> ---
        {
            HighFive::File file("test/test_no_metadata.h5", HighFive::File::ReadOnly);
            EXPECT_FALSE(file.hasAttribute("time"));

            pugi::xml_document doc;
            ASSERT_TRUE(doc.load_file("test/test_no_metadata.xdmf"));
            auto domain = doc.child("Xdmf").child("Domain");
            for (pugi::xml_node grid : domain.children("Grid"))
            {
                EXPECT_TRUE(grid.child("Time").empty());
            }
        }
#endif
    }
}
