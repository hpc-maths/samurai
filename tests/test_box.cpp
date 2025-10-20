#include <algorithm>

#include <gtest/gtest.h>
// #include <rapidcheck/gtest.h>

#include <xtensor/containers/xarray.hpp>

#include <samurai/box.hpp>

namespace samurai
{

    // RC_GTEST_PROP(Box, corner,
    //               (std::array<int, 2> min,
    //                std::array<int, 2> max))
    // {
    //     RC_PRE(min[0] < max[0]);
    //     RC_PRE(min[1] < max[1]);

    //     Box<int, 2> box{{min[0], min[1]}, {max[0], max[1]}};
    //     xt::xarray<int> expected{min[0], min[1]};
    //     RC_ASSERT(box.min_corner() == expected);
    // }

    TEST(box, min_corner)
    {
        Box<int, 2> box{
            {0, 0},
            {1, 1}
        };
        xt::xarray<int> expected{0, 0};
        EXPECT_EQ(box.min_corner(), expected);
    }

    TEST(box, max_corner)
    {
        Box<int, 2> box{
            {0, 0},
            {1, 1}
        };
        xt::xarray<int> expected{1, 1};
        EXPECT_EQ(box.max_corner(), expected);
    }

    TEST(box, length)
    {
        Box<int, 2> box_1{
            {0,  0},
            {10, 5}
        };
        xt::xarray<int> expected_1{10, 5};

        Box<int, 2> box_2{
            {-5, -10},
            {10, 5  }
        };
        xt::xarray<int> expected_2{15, 15};

        EXPECT_EQ(box_1.length(), expected_1);
        EXPECT_EQ(box_2.length(), expected_2);
    }

    TEST(box, is_valid)
    {
        EXPECT_TRUE((Box<int, 2>{
                         {0, 0},
                         {1, 1}
        })
                        .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {1, 1},
                          {0, 0}
        })
                         .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {0, 0},
                          {1, 0}
        })
                         .is_valid());
        EXPECT_FALSE((Box<int, 2>{
                          {0, 0},
                          {0, 1}
        })
                         .is_valid());
    }

    TEST(box, operator)
    {
        Box<int, 2> b{
            {-1, -1},
            {1,  1 }
        };
        xt::xarray<int> expected_min{-5, -5};
        xt::xarray<int> expected_max{5, 5};

        auto bl = b * 5;
        auto br = 5 * b;
        EXPECT_EQ(bl.min_corner(), expected_min);
        EXPECT_EQ(br.min_corner(), expected_min);
        EXPECT_EQ(bl.max_corner(), expected_max);
        EXPECT_EQ(br.max_corner(), expected_max);
        EXPECT_EQ(bl.length(), 5 * b.length());
        EXPECT_EQ(br.length(), 5 * b.length());
    }

    TEST(box, ostream)
    {
        Box<int, 2> b{
            {-1, -1},
            {1,  1 }
        };
        std::stringstream ss;
        ss << b;
        EXPECT_STREQ(ss.str().data(), "Box({-1, -1}, {1, 1})");
    }

    TEST(box, approximate_box_exact)
    {
        Box<double, 2> box{
            {-1., -1.},
            {1.,  1. }
        };
        // The approximation should be exact
        double tol                = 0.5; // even with a large tolerance
        double subdivision_length = -1;
        auto approx_box           = approximate_box(box, tol, subdivision_length);
        EXPECT_EQ(approx_box, box); // exact approximation
        EXPECT_EQ(subdivision_length, 2.);
    }

    TEST(box, approximate_box_exact_with_gcd)
    {
        Box<double, 2> box{
            {0.,   0. },
            {1.25, 0.5}
        };
        // The approximation should be exact
        double tol                = 0.2; // even with a tolerance
        double subdivision_length = -1;
        auto approx_box           = approximate_box(box, tol, subdivision_length);
        EXPECT_EQ(approx_box, box);          // exact approximation
        EXPECT_EQ(subdivision_length, 0.25); // Greatest Common Divisor (GCD) of 1.25 and 0.5
    }

    TEST(box, approximate_box_with_tolerance)
    {
        Box<double, 2> box{
            {0.,  0.  },
            {0.8, 1.61}
        };
        // With no tolerance, the approximation should be exact.
        double tol                = 0.;
        double subdivision_length = -1;
        auto approx_box           = approximate_box(box, tol, subdivision_length);
        EXPECT_EQ(approx_box, box);          // exact approximation
        EXPECT_EQ(subdivision_length, 0.01); // Greatest Common Divisor (GCD) of 0.8 and 1.61

        // With a tolerance, the algorithm finds the largest subdivision that fits the tolerance.
        tol                        = 0.05;
        double subdivision_length2 = -1;
        approx_box                 = approximate_box(box, tol, subdivision_length2);
        EXPECT_GT(subdivision_length2, subdivision_length);                                      // subdivision_length2 > subdivision_length
        EXPECT_TRUE(xt::all(xt::abs(approx_box.length() - box.length()) <= tol * box.length())); // the approximation fits the tolerance
    }

    TEST(box, approximate_box_exact_no_admissible_gcd)
    {
        Box<double, 2> box{
            {0.,   0.              },
            {1.25, 0.55551111111111}
        };
        // The GCD of the lengths is too small to be not admissible.
        // A tolerance is required to approximate the box.
        double tol                = 0.02;
        double subdivision_length = -1;
        auto approx_box           = approximate_box(box, tol, subdivision_length);
        EXPECT_TRUE(xt::all(xt::abs(approx_box.length() - box.length()) <= tol * box.length())); // the approximation fits the tolerance
    }

    /**
     * The box to remove is fully inside the box.
     */
    TEST(box, difference_inside_2D)
    {
        static constexpr std::size_t dim = 2;
        Box<double, dim> box{
            {-1., -1.},
            {1.,  1. }
        };
        Box<double, dim> box_to_remove{
            {-0.5, -0.5},
            {0.5,  0.5 }
        };
        auto boxes = box.difference(box_to_remove);
        EXPECT_EQ(boxes.size(), std::pow(3, dim) - 1);
    }

    /**
     * The box to remove is fully inside the box.
     * Same test in 3D.
     */
    TEST(box, difference_inside_3D)
    {
        static constexpr std::size_t dim = 3;
        Box<double, dim> box{
            {-1., -1., -1.},
            {1.,  1.,  1. }
        };
        Box<double, dim> box_to_remove{
            {-0.5, -0.5, -0.5},
            {0.5,  0.5,  0.5 }
        };
        auto boxes = box.difference(box_to_remove);
        EXPECT_EQ(boxes.size(), std::pow(3, dim) - 1);
    }

    /**
     * The box to remove is at a corner of the box.
     */
    TEST(box, difference_in_corner)
    {
        static constexpr std::size_t dim = 2;
        Box<double, dim> box{
            {-1., -1.},
            {1.,  1. }
        };
        Box<double, dim> box_to_remove{
            {-1., -1.},
            {0.5, 0.5}
        };
        auto boxes = box.difference(box_to_remove);
        EXPECT_EQ(boxes.size(), 3);
    }

    /***
     * The box to remove is partly inside and partly outside the box.
     */
    TEST(box, difference_overlap)
    {
        static constexpr std::size_t dim = 2;
        Box<double, dim> box{
            {-1., -1.},
            {1.,  1. }
        };
        Box<double, dim> box_to_remove{
            {-2., -2.},
            {0.5, 0.5}
        };
        auto boxes = box.difference(box_to_remove);
        EXPECT_EQ(boxes.size(), 3);
    }

    /**
     * The box to remove has no intersection with the box.
     */
    TEST(box, difference_outside)
    {
        static constexpr std::size_t dim = 2;
        Box<double, dim> box{
            {-1., -1.},
            {1.,  1. }
        };
        Box<double, dim> box_to_remove{
            {-2., -2.},
            {-1., -1.}
        };
        auto boxes = box.difference(box_to_remove);
        EXPECT_EQ(boxes.size(), 1);
        EXPECT_EQ(boxes[0], box);
    }

}
