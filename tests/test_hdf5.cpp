#include <map>

#include <gtest/gtest.h>

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
        save("test_save_mesh.h5", mesh);
        save("test", "test_save_mesh.h5", mesh);
        save("test", "test_save_mesh.h5", {true, true}, mesh);
        save("test_save_mesh.h5", {true, true}, mesh);
    }

    template <typename config_t>
    void test_save_uniform(const UniformMesh<config_t>& mesh)
    {
        save("test_save_mesh.h5", mesh);
        save("test", "test_save_mesh.h5", mesh);
        save("test", "test_save_mesh.h5", {true}, mesh);
        save("test_save_mesh.h5", {true}, mesh);
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
        using Config                     = MRConfig<dim>;
        using Mesh                       = MRMesh<Config>;
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);
        Box<double, dim> box(min_corner, max_corner);
        auto mesh_cfg = samurai::mesh_config<dim>().min_level(4).max_level(4);
        Mesh mesh(mesh_cfg, box);
        test_save(mesh);
        args::save_debug_fields = true;
        test_save(mesh);
    }
}
