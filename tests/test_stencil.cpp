#include <gtest/gtest.h>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{
    template <typename T>
    class stencil_test : public ::testing::Test
    {
    };

    using stencil_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(stencil_test, stencil_test_types, );

    TYPED_TEST(stencil_test, convert_for_direction)
    {
        static constexpr std::size_t dim = TypeParam::value;

        // Stencil in the x direction
        constexpr std::size_t x_direction  = 0;
        constexpr std::size_t stencil_size = 2;
        auto stencil_in_x                  = samurai::line_stencil<dim, x_direction, stencil_size>();

        // Check that the second vector of the rotated stencil corresponds to the direction
        // (the first vector is 0, the stencil center)
        samurai::for_each_cartesian_direction<dim>(
            [&](const auto& direction)
            {
                auto rotated_stencil = samurai::convert_for_direction(stencil_in_x, direction);

                samurai::DirectionVector<dim> rotated_dir = xt::view(rotated_stencil, 1);
                EXPECT_EQ(rotated_dir, direction);
            });

        samurai::for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                auto rotated_stencil = samurai::convert_for_direction(stencil_in_x, direction);

                samurai::DirectionVector<dim> rotated_dir = xt::view(rotated_stencil, 1);
                EXPECT_EQ(rotated_dir, direction);
            });
    }
}
