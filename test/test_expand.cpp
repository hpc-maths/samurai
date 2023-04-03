#include <gtest/gtest.h>

#include <samurai/interval.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/subset/subset_op.hpp>
#include <samurai/hdf5.hpp>

namespace samurai
{
    template <typename T>
    class expand_test : public ::testing::Test{};

    using dim_test_types = ::testing::Types<
    std::integral_constant<std::size_t, 1>,
    std::integral_constant<std::size_t, 2>,
    std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(expand_test, dim_test_types);

    TYPED_TEST(expand_test, with_size)
    {
        static constexpr std::size_t dim = TypeParam::value;
        using box_t = Box<int, dim>;
        using lca_t = LevelCellArray<dim>;
        std::size_t size = 2;

        xt::xtensor_fixed<int, xt::xshape<dim>> min_corner, max_corner;
        min_corner.fill(0);
        max_corner.fill(1);
        lca_t lca_i{1, box_t{min_corner, max_corner}};

        xt::xtensor_fixed<int, xt::xshape<dim>> min_corner_e, max_corner_e;
        min_corner_e.fill(-size);
        max_corner_e.fill(size+1);
        lca_t lca_e{1, box_t{min_corner_e, max_corner_e}};

        xt::xtensor_fixed<int, xt::xshape<dim>> t;
        t.fill(0);
        t[0] = 1;
        auto set = translate(expand(lca_i, size), t);
        lca_t lca(intersection(set, set));
        lca.update_index();

        auto set2 = expand(translate(lca_i, t), size);
        lca_t lca2(intersection(set2, set2));
        lca2.update_index();
        EXPECT_EQ(lca, lca_e);
        EXPECT_EQ(lca, lca2);
    }


    TEST(expand, to_fix)
    {
        constexpr std::size_t dim = 2;
        using lcl_t = LevelCellList<dim>;
        using lca_t = LevelCellArray<dim>;

        std::size_t size = 1;

        lcl_t lcl(0);
        lcl[{0}].add_interval({0, 3});
        // lcl[{1}].add_interval({1, 2});
        lcl[{1}].add_interval({4, 5});
        // lcl[{-1}].add_interval({1, 2});
        lca_t mesh{lcl};
        lca_t lca = intersection(expand(mesh, size), expand(mesh, size));
        // lca_t lca = intersection(mesh, translate(mesh, stencil));
        std::cout << lca << std::endl;
        save("mesh_init", mesh);
        save("mesh", lca);
    }
    // TYPED_TEST(expand_test, with_dir)
    // {
    //     static constexpr std::size_t dim = TypeParam::value;
    //     using box_t = Box<int, dim>;
    //     using lca_t = LevelCellArray<dim>;
    //     std::size_t size = 2;

    //     xt::xtensor_fixed<int, xt::xshape<dim>> min_corner, max_corner;
    //     min_corner.fill(0);
    //     max_corner.fill(1);
    //     lca_t lca_i{1, box_t{min_corner, max_corner}};

    //     xt::xtensor_fixed<int, xt::xshape<dim>> min_corner_e, max_corner_e;
    //     min_corner_e.fill(0);
    //     max_corner_e.fill(size+1);
    //     lca_t lca_e{1, box_t{min_corner_e, max_corner_e}};

    //     xt::xtensor_fixed<int, xt::xshape<dim>> dir;
    //     dir.fill(1);
    //     lca_t lca(expand(lca_i, dir, size));
    //     lca.update_index();
    //     EXPECT_EQ(lca, lca_e);

    //     min_corner_e.fill(-size);
    //     max_corner_e.fill(1);
    //     lca_e = {1, box_t{min_corner_e, max_corner_e}};
    //     dir.fill(-1);
    //     lca = expand(lca_i, dir, size);
    //     lca.update_index();
    //     EXPECT_EQ(lca, lca_e);
    // }

}