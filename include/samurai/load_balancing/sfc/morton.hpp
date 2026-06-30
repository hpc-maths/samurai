// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Morton (Z-order) curve: the key interleaves the bits of the coordinates
 * (bit b of coordinate d lands at key bit b*dim + d).
 *
 * Properties: O(1) key computation with pure bit tricks, good (not optimal)
 * locality — the curve jumps at power-of-two boundaries, unlike Hilbert.
 * Range: 32 bits per coordinate in 2D, 21 bits in 3D (key fits in 64 bits).
 * The inverse mapping is provided (`decode_2d` / `decode_3d`).
 */

#include <cassert>
#include <cstdint>

#include <xtensor/containers/xfixed.hpp>

#include "sfc.hpp"

namespace samurai::load_balancing
{
    class Morton : public SFCCurve<Morton>
    {
      public:

        std::string name_impl() const
        {
            return "morton";
        }

        /// 2D key: bits of i at even positions, bits of j at odd positions.
        template <class Coord>
        sfc_key_t key_2d(const Coord& p) const
        {
            return split2(p(0)) | (split2(p(1)) << 1);
        }

        /// 3D key: bits of (i, j, k) at positions (3b, 3b+1, 3b+2).
        template <class Coord>
        sfc_key_t key_3d(const Coord& p) const
        {
            assert(p(0) <= 0x1fffff && p(1) <= 0x1fffff && p(2) <= 0x1fffff && "Morton 3D supports 21 bits per coordinate");
            return split3(p(0)) | (split3(p(1)) << 1) | (split3(p(2)) << 2);
        }

        /// Rectangle-aware overloads: Morton interleaving is independent of the
        /// bounding box, so the extent `n` is ignored (the curve is allowed to
        /// be spatially disconnected -- only Hilbert exploits `n`).
        template <class Coord, class Extent>
        sfc_key_t key_2d(const Coord& p, const Extent& /*n*/) const
        {
            return key_2d(p);
        }

        template <class Coord, class Extent>
        sfc_key_t key_3d(const Coord& p, const Extent& /*n*/) const
        {
            return key_3d(p);
        }

        /// Inverse of key_2d (Morton-specific extra, used for debug/tests).
        static auto decode_2d(sfc_key_t key)
        {
            xt::xtensor_fixed<std::uint32_t, xt::xshape<2>> p;
            p(0) = compact2(key);
            p(1) = compact2(key >> 1);
            return p;
        }

        /// Inverse of key_3d.
        static auto decode_3d(sfc_key_t key)
        {
            xt::xtensor_fixed<std::uint32_t, xt::xshape<3>> p;
            p(0) = compact3(key);
            p(1) = compact3(key >> 1);
            p(2) = compact3(key >> 2);
            return p;
        }

      private:

        /// Spread the 32 bits of x over the even bits of a 64-bit word.
        static sfc_key_t split2(std::uint32_t x)
        {
            sfc_key_t v = x;
            v           = (v | (v << 16)) & 0x0000ffff0000ffff;
            v           = (v | (v << 8)) & 0x00ff00ff00ff00ff;
            v           = (v | (v << 4)) & 0x0f0f0f0f0f0f0f0f;
            v           = (v | (v << 2)) & 0x3333333333333333;
            v           = (v | (v << 1)) & 0x5555555555555555;
            return v;
        }

        /// Spread the 21 low bits of x every 3 bits of a 64-bit word.
        static sfc_key_t split3(std::uint32_t x)
        {
            sfc_key_t v = x & 0x1fffff;
            v           = (v | (v << 32)) & 0x001f00000000ffff;
            v           = (v | (v << 16)) & 0x001f0000ff0000ff;
            v           = (v | (v << 8)) & 0x100f00f00f00f00f;
            v           = (v | (v << 4)) & 0x10c30c30c30c30c3;
            v           = (v | (v << 2)) & 0x1249249249249249;
            return v;
        }

        static std::uint32_t compact2(sfc_key_t v)
        {
            v = v & 0x5555555555555555;
            v = (v ^ (v >> 1)) & 0x3333333333333333;
            v = (v ^ (v >> 2)) & 0x0f0f0f0f0f0f0f0f;
            v = (v ^ (v >> 4)) & 0x00ff00ff00ff00ff;
            v = (v ^ (v >> 8)) & 0x0000ffff0000ffff;
            v = (v ^ (v >> 16)) & 0x00000000ffffffff;
            return static_cast<std::uint32_t>(v);
        }

        static std::uint32_t compact3(sfc_key_t v)
        {
            v = v & 0x1249249249249249;
            v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3;
            v = (v ^ (v >> 4)) & 0x100f00f00f00f00f;
            v = (v ^ (v >> 8)) & 0x001f0000ff0000ff;
            v = (v ^ (v >> 16)) & 0x001f00000000ffff;
            v = (v ^ (v >> 32)) & 0x1fffff;
            return static_cast<std::uint32_t>(v);
        }
    };
}
