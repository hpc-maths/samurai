// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include <samurai/static_dispatch.hpp>

namespace samurai
{
    // The callable receives the runtime value as a compile-time constant.
    TEST(static_dispatch, maps_value_to_compile_time_constant)
    {
        for (std::size_t value = 1; value <= 4; ++value)
        {
            std::size_t seen = dispatch_static<1, 4>(value,
                                                     [](auto ic)
                                                     {
                                                         static constexpr std::size_t N = decltype(ic)::value;
                                                         return N;
                                                     });
            EXPECT_EQ(seen, value);
        }
    }

    // Both bounds are inclusive.
    TEST(static_dispatch, bounds_are_inclusive)
    {
        // Extra parentheses: the comma in the template argument list must not be
        // parsed as a macro-argument separator.
        EXPECT_NO_THROW((dispatch_static<2, 5>(2, [](auto) {})));
        EXPECT_NO_THROW((dispatch_static<2, 5>(5, [](auto) {})));
    }

    // A value outside [min, max] throws.
    TEST(static_dispatch, out_of_range_throws)
    {
        EXPECT_THROW((dispatch_static<2, 5>(1, [](auto) {})), std::out_of_range);
        EXPECT_THROW((dispatch_static<2, 5>(6, [](auto) {})), std::out_of_range);
        EXPECT_THROW((dispatch_static<2, 5>(0, [](auto) {})), std::out_of_range);
    }

    // The return value is forwarded to the caller.
    TEST(static_dispatch, forwards_return_value)
    {
        auto squared = dispatch_static<1, 3>(3,
                                             [](auto ic)
                                             {
                                                 static constexpr std::size_t N = decltype(ic)::value;
                                                 return N * N;
                                             });
        EXPECT_EQ(squared, 9u);
    }

    // A void-returning callable is supported.
    TEST(static_dispatch, supports_void_callable)
    {
        std::size_t captured = 0;
        dispatch_static<0, 3>(2,
                              [&](auto ic)
                              {
                                  captured = decltype(ic)::value;
                              });
        EXPECT_EQ(captured, 2u);
    }

    // The callable is invoked exactly once, only for the matching value.
    TEST(static_dispatch, invoked_exactly_once)
    {
        int calls = 0;
        dispatch_static<1, 6>(4,
                              [&](auto)
                              {
                                  ++calls;
                              });
        EXPECT_EQ(calls, 1);
    }

    // A single-value range [N, N] is valid.
    TEST(static_dispatch, single_value_range)
    {
        std::size_t seen = dispatch_static<7, 7>(7,
                                                 [](auto ic)
                                                 {
                                                     return decltype(ic)::value;
                                                 });
        EXPECT_EQ(seen, 7u);
        EXPECT_THROW((dispatch_static<7, 7>(8, [](auto) {})), std::out_of_range);
    }

    // The compile-time constant can drive a template parameter / array size.
    TEST(static_dispatch, drives_template_parameter)
    {
        auto sum = dispatch_static<1, 4>(3,
                                         [](auto ic)
                                         {
                                             static constexpr std::size_t N = decltype(ic)::value;
                                             std::array<int, N> a{};
                                             a.fill(2);
                                             int s = 0;
                                             for (auto v : a)
                                             {
                                                 s += v;
                                             }
                                             return s;
                                         });
        EXPECT_EQ(sum, 6); // 3 elements * 2
    }
}
