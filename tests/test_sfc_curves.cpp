// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Space-filling curves of the load balancing module (roadmap step 3).
// MPI-free: the curves are pure functions of the coordinates.

#include <cstdint>
#include <map>

#include <gtest/gtest.h>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/load_balancing/sfc/hilbert.hpp>
#include <samurai/load_balancing/sfc/morton.hpp>

namespace lb = samurai::load_balancing;

namespace
{
    using coord2 = xt::xtensor_fixed<std::uint32_t, xt::xshape<2>>;
    using coord3 = xt::xtensor_fixed<std::uint32_t, xt::xshape<3>>;

    TEST(sfc_morton, reference_values_2d)
    {
        const lb::Morton morton;
        // Z-order: bit b of i at key bit 2b, bit b of j at key bit 2b+1
        EXPECT_EQ(morton.key<2>(coord2{0, 0}), 0U);
        EXPECT_EQ(morton.key<2>(coord2{1, 0}), 1U);
        EXPECT_EQ(morton.key<2>(coord2{0, 1}), 2U);
        EXPECT_EQ(morton.key<2>(coord2{1, 1}), 3U);
        EXPECT_EQ(morton.key<2>(coord2{2, 0}), 4U);
        EXPECT_EQ(morton.key<2>(coord2{0, 2}), 8U);
        EXPECT_EQ(morton.key<2>(coord2{5, 3}), 27U); // 101 ⨉ 011 -> 011011
    }

    TEST(sfc_morton, reference_values_3d)
    {
        const lb::Morton morton;
        EXPECT_EQ(morton.key<3>(coord3{1, 0, 0}), 1U);
        EXPECT_EQ(morton.key<3>(coord3{0, 1, 0}), 2U);
        EXPECT_EQ(morton.key<3>(coord3{0, 0, 1}), 4U);
        EXPECT_EQ(morton.key<3>(coord3{1, 1, 1}), 7U);
        EXPECT_EQ(morton.key<3>(coord3{2, 2, 2}), 56U);
    }

    TEST(sfc_morton, roundtrip_2d_exhaustive)
    {
        const lb::Morton morton;
        for (std::uint32_t j = 0; j < 256; ++j)
        {
            for (std::uint32_t i = 0; i < 256; ++i)
            {
                const auto key = morton.key<2>(coord2{i, j});
                const auto p   = lb::Morton::decode_2d(key);
                ASSERT_EQ(p(0), i);
                ASSERT_EQ(p(1), j);
            }
        }
    }

    TEST(sfc_morton, roundtrip_3d_exhaustive)
    {
        const lb::Morton morton;
        for (std::uint32_t k = 0; k < 64; ++k)
        {
            for (std::uint32_t j = 0; j < 64; ++j)
            {
                for (std::uint32_t i = 0; i < 64; ++i)
                {
                    const auto key = morton.key<3>(coord3{i, j, k});
                    const auto p   = lb::Morton::decode_3d(key);
                    ASSERT_EQ(p(0), i);
                    ASSERT_EQ(p(1), j);
                    ASSERT_EQ(p(2), k);
                }
            }
        }
    }

    TEST(sfc_morton, full_range_corners)
    {
        const lb::Morton morton;
        // 2D: 32 bits per coordinate fill the 64 key bits
        const std::uint32_t max2 = 0xffffffffU;
        EXPECT_EQ(morton.key<2>(coord2{max2, max2}), 0xffffffffffffffffULL);
        const auto p2 = lb::Morton::decode_2d(0xffffffffffffffffULL);
        EXPECT_EQ(p2(0), max2);
        EXPECT_EQ(p2(1), max2);
        // 3D: 21 bits per coordinate fill 63 key bits
        const std::uint32_t max3 = 0x1fffffU;
        EXPECT_EQ(morton.key<3>(coord3{max3, max3, max3}), 0x7fffffffffffffffULL);
    }

    // The defining property of Hilbert (and the non-regression test of the
    // historic 3D overflow): on any dyadic block, the sorted keys are
    // consecutive integers and two consecutive keys are face-adjacent cells
    // (Manhattan distance exactly 1). Morton fails this (it jumps).
    template <std::size_t dim>
    void check_hilbert_block(std::uint32_t base, std::uint32_t n)
    {
        const lb::Hilbert hilbert;
        std::map<lb::sfc_key_t, std::array<std::uint32_t, dim>> cells;

        // loops on offsets: `base + n` may overflow uint32 when the window
        // touches the top of the coordinate range
        if constexpr (dim == 2)
        {
            for (std::uint32_t dj = 0; dj < n; ++dj)
            {
                for (std::uint32_t di = 0; di < n; ++di)
                {
                    const std::uint32_t i = base + di;
                    const std::uint32_t j = base + dj;

                    cells[hilbert.key<2>(coord2{i, j})] = {i, j};
                }
            }
        }
        else
        {
            for (std::uint32_t dk = 0; dk < n; ++dk)
            {
                for (std::uint32_t dj = 0; dj < n; ++dj)
                {
                    for (std::uint32_t di = 0; di < n; ++di)
                    {
                        const std::uint32_t i = base + di;
                        const std::uint32_t j = base + dj;
                        const std::uint32_t k = base + dk;

                        cells[hilbert.key<3>(coord3{i, j, k})] = {i, j, k};
                    }
                }
            }
        }

        // injectivity: as many distinct keys as cells
        std::size_t expected = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            expected *= n;
        }
        ASSERT_EQ(cells.size(), expected);

        // continuity: consecutive keys, Manhattan distance 1
        auto prev = cells.begin();
        for (auto it = std::next(cells.begin()); it != cells.end(); ++it, ++prev)
        {
            ASSERT_EQ(it->first, prev->first + 1) << "keys of a dyadic block must be consecutive";
            std::uint64_t manhattan = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                manhattan += (it->second[d] > prev->second[d]) ? it->second[d] - prev->second[d] : prev->second[d] - it->second[d];
            }
            ASSERT_EQ(manhattan, 1U) << "consecutive Hilbert keys must be face-adjacent";
        }
    }

    TEST(sfc_hilbert, continuity_2d)
    {
        check_hilbert_block<2>(0, 64);
    }

    TEST(sfc_hilbert, continuity_3d)
    {
        check_hilbert_block<3>(0, 16);
    }

    // High-bit windows: top dyadic blocks of the coordinate range. The 3D
    // case would have produced corrupted keys before the 64/dim bit fix.
    TEST(sfc_hilbert, continuity_2d_high_bits)
    {
        check_hilbert_block<2>(0xffffffffU - 63U, 64); // [2^32-64, 2^32)
    }

    TEST(sfc_hilbert, continuity_3d_high_bits)
    {
        check_hilbert_block<3>((1U << 21) - 16U, 16); // [2^21-16, 2^21)
    }

    // Rectangle-aware Hilbert (generalized "gilbert" curve): over an arbitrary
    // w x h box the keys must be a bijection onto [0, w*h) AND consecutive keys
    // must be face-adjacent. The continuity over a *non-square* box is what the
    // square curve lacks and what keeps a load-balancing partition spatially
    // connected on thin domains (e.g. a [1 x 10] tube).
    void check_gilbert_rect(std::uint32_t w, std::uint32_t h)
    {
        const lb::Hilbert hilbert;
        const coord2 n{w, h};
        std::map<lb::sfc_key_t, coord2> cells;
        for (std::uint32_t j = 0; j < h; ++j)
        {
            for (std::uint32_t i = 0; i < w; ++i)
            {
                cells[hilbert.key<2>(coord2{i, j}, n)] = coord2{i, j};
            }
        }

        const std::size_t expected_count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        ASSERT_EQ(cells.size(), expected_count) << "gilbert keys must be a bijection on " << w << "x" << h;

        lb::sfc_key_t expected_key = 0;
        bool has_prev              = false;
        coord2 prev{0, 0};
        for (const auto& [key, p] : cells)
        {
            ASSERT_EQ(key, expected_key) << "gilbert keys must be the consecutive integers 0.." << expected_count - 1;
            if (has_prev)
            {
                const std::uint64_t dx = (p(0) > prev(0)) ? p(0) - prev(0) : prev(0) - p(0);
                const std::uint64_t dy = (p(1) > prev(1)) ? p(1) - prev(1) : prev(1) - p(1);
                ASSERT_EQ(dx + dy, 1U) << "consecutive gilbert keys must be face-adjacent on " << w << "x" << h;
            }
            prev     = p;
            has_prev = true;
            ++expected_key;
        }
    }

    TEST(sfc_hilbert, gilbert_rectangles)
    {
        check_gilbert_rect(1, 1);
        check_gilbert_rect(1, 10); // thin tube (degenerate width)
        check_gilbert_rect(10, 1);
        check_gilbert_rect(4, 10);
        check_gilbert_rect(8, 8); // square, power of two
        check_gilbert_rect(7, 7); // square, odd
        check_gilbert_rect(3, 7);
        check_gilbert_rect(13, 5);
        check_gilbert_rect(32, 320); // tube discretized at level 5
    }

    // On a square power-of-two box the generalized curve must keep the defining
    // Hilbert property (consecutive keys face-adjacent); only the index origin
    // may differ from key<2>(p). Covered by gilbert_rectangles(8x8); this is the
    // explicit non-regression marker.
    TEST(sfc_hilbert, gilbert_square_is_continuous)
    {
        check_gilbert_rect(64, 64);
    }

    TEST(sfc_curves, dim1_is_identity)
    {
        const lb::Morton morton;
        const lb::Hilbert hilbert;
        const xt::xtensor_fixed<std::uint32_t, xt::xshape<1>> p{42};
        EXPECT_EQ(morton.key<1>(p), 42U);
        EXPECT_EQ(hilbert.key<1>(p), 42U);
    }
}
