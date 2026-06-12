// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Hilbert curve via Skilling's transposition algorithm.
 *
 * Reference: J. Skilling, "Programming the Hilbert curve",
 * AIP Conference Proceedings 707, 381 (2004).
 *
 * The coordinates are first transposed in place (Gray-code based pass over
 * the bit planes), then the transposed bits are interleaved into the key
 * exactly like Morton. Unlike Morton, the Hilbert curve is *continuous*:
 * two consecutive keys are always face-adjacent cells (Manhattan distance 1),
 * which gives the best locality of all practical SFCs.
 *
 * Range: `max_bits(dim)` bits per coordinate so the key fits in 64 bits —
 * 32 bits in 2D, 21 bits in 3D (deepest usable level: 21 in 3D). The historic
 * implementation used 32 bits per coordinate in all dimensions and overflowed
 * the 64-bit key in 3D (`1 << 95`); bounding the bit count both fixes the
 * overflow and shortens the transposition loop.
 *
 * No inverse mapping: the partitioning only needs coordinates -> key.
 */

#include <cassert>
#include <cstdint>

#include <xtensor/containers/xfixed.hpp>

#include "sfc.hpp"

namespace samurai::load_balancing
{
    class Hilbert : public SFCCurve<Hilbert>
    {
      public:

        std::string name() const
        {
            return "hilbert";
        }

        template <class Coord>
        sfc_key_t key_2d(const Coord& p) const
        {
            return key_impl<2>(p);
        }

        template <class Coord>
        sfc_key_t key_3d(const Coord& p) const
        {
            return key_impl<3>(p);
        }

      private:

        template <std::size_t dim, class Coord>
        sfc_key_t key_impl(const Coord& p) const
        {
            constexpr unsigned nbits = max_bits(dim);

            xt::xtensor_fixed<std::uint32_t, xt::xshape<dim>> x;
            for (std::size_t d = 0; d < dim; ++d)
            {
                assert((nbits == 32U || p(d) < (1U << nbits)) && "coordinate exceeds the usable Hilbert range");
                x(d) = p(d);
            }

            // Skilling: axes -> transposed Hilbert coordinates (in place)
            for (std::uint32_t q = 1U << (nbits - 1); q > 1; q >>= 1)
            {
                const std::uint32_t mask = q - 1;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (x(d) & q)
                    {
                        x(0) ^= mask; // invert low bits of x(0)
                    }
                    else
                    {
                        const std::uint32_t t = (x(0) ^ x(d)) & mask;
                        x(0) ^= t;
                        x(d) ^= t;
                    }
                }
            }

            // Gray decode
            for (std::size_t d = 1; d < dim; ++d)
            {
                x(d) ^= x(d - 1);
            }
            std::uint32_t t = 0;
            for (std::uint32_t q = 1U << (nbits - 1); q > 1; q >>= 1)
            {
                if (x(dim - 1) & q)
                {
                    t ^= q - 1;
                }
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                x(d) ^= t;
            }

            // Interleave the transposed bits into the key (x(0) carries the
            // most significant bit), bounded by nbits*dim <= 64.
            sfc_key_t key = 0;
            int shift     = static_cast<int>(nbits * dim) - 1;
            for (int b = static_cast<int>(nbits) - 1; b >= 0; --b)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    key |= static_cast<sfc_key_t>((x(d) >> b) & 1U) << shift;
                    --shift;
                }
            }
            return key;
        }
    };
}
