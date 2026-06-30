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
 * Rectangular domains (2D): the square `key_2d(p)` above maps a 2^k x 2^k grid;
 * restricted to a thin strip its locality breaks (the curve leaves and re-enters
 * the strip, so a contiguous arc is several disjoint pieces -> a load-balancing
 * partition fractures into spatial islands). `key_2d(p, n)` instead lays a
 * *generalized* Hilbert curve (Jakub Cerveny's "gilbert") over the exact
 * `n(0) x n(1)` bounding box: a single continuous curve with matched seams that
 * fills an arbitrary rectangle, so contiguous arcs stay spatially connected. On
 * a square power-of-two box it coincides with the standard Hilbert order.
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

        std::string name_impl() const
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

        /// Rectangle-aware 2D key: generalized Hilbert curve over the
        /// `n(0) x n(1)` bounding box (see the file header). Coincides with
        /// `key_2d(p)` on a square power-of-two box.
        template <class Coord, class Extent>
        sfc_key_t key_2d(const Coord& p, const Extent& n) const
        {
            const std::int64_t w = static_cast<std::int64_t>(n(0));
            const std::int64_t h = static_cast<std::int64_t>(n(1));
            const std::int64_t x = static_cast<std::int64_t>(p(0));
            const std::int64_t y = static_cast<std::int64_t>(p(1));
            assert(w >= 1 && h >= 1 && x >= 0 && x < w && y >= 0 && y < h && "point outside the gilbert bounding box");
            // major axis along the longer side
            if (w >= h)
            {
                return gilbert_xy2d(x, y, 0, 0, w, 0, 0, h);
            }
            return gilbert_xy2d(x, y, 0, 0, 0, h, w, 0);
        }

        /// Rectangle-aware 3D key: the generalized 3D Hilbert curve is not
        /// implemented; fall back to the square curve (correct, but with the
        /// thin-domain locality caveat for strongly non-cubic 3D boxes).
        template <class Coord, class Extent>
        sfc_key_t key_3d(const Coord& p, const Extent& /*n*/) const
        {
            return key_impl<3>(p);
        }

      private:

        static std::int64_t sgn(std::int64_t v)
        {
            return static_cast<std::int64_t>(v > 0) - static_cast<std::int64_t>(v < 0);
        }

        static std::int64_t iabs(std::int64_t v)
        {
            return v < 0 ? -v : v;
        }

        /// Is (px, py) inside the axis-aligned region originating at (x, y) and
        /// spanned by A = (ax, ay) and B = (bx, by)? In the gilbert recursion A
        /// and B are always orthogonal and axis-aligned, so the region is the
        /// |A| x |B| integer box with (x, y) as its near corner.
        static bool in_region(std::int64_t px,
                              std::int64_t py,
                              std::int64_t x,
                              std::int64_t y,
                              std::int64_t ax,
                              std::int64_t ay,
                              std::int64_t bx,
                              std::int64_t by)
        {
            const std::int64_t ex  = ax + bx; // signed x-extent (one of ax/bx is 0)
            const std::int64_t ey  = ay + by; // signed y-extent (one of ay/by is 0)
            const std::int64_t fx  = x + ex - sgn(ex);
            const std::int64_t fy  = y + ey - sgn(ey);
            const std::int64_t xlo = std::min(x, fx), xhi = std::max(x, fx);
            const std::int64_t ylo = std::min(y, fy), yhi = std::max(y, fy);
            return px >= xlo && px <= xhi && py >= ylo && py <= yhi;
        }

        /// Distance along the generalized Hilbert curve of point (x_dst, y_dst)
        /// in the region originating at (x, y), major direction A = (ax, ay),
        /// orthogonal direction B = (bx, by). Cerveny's algorithm.
        static sfc_key_t gilbert_xy2d(std::int64_t x_dst,
                                      std::int64_t y_dst,
                                      std::int64_t x,
                                      std::int64_t y,
                                      std::int64_t ax,
                                      std::int64_t ay,
                                      std::int64_t bx,
                                      std::int64_t by)
        {
            const std::int64_t w = iabs(ax + ay);
            const std::int64_t h = iabs(bx + by);

            const std::int64_t dax = sgn(ax), day = sgn(ay); // major unit direction
            const std::int64_t dbx = sgn(bx), dby = sgn(by); // orthogonal unit direction

            if (h == 1) // trivial row: fill along A
            {
                return static_cast<sfc_key_t>(dax * (x_dst - x) + day * (y_dst - y));
            }
            if (w == 1) // trivial column: fill along B
            {
                return static_cast<sfc_key_t>(dbx * (x_dst - x) + dby * (y_dst - y));
            }

            std::int64_t ax2 = ax / 2, ay2 = ay / 2;
            std::int64_t bx2 = bx / 2, by2 = by / 2;

            const std::int64_t w2 = iabs(ax2 + ay2);
            const std::int64_t h2 = iabs(bx2 + by2);

            if (2 * w > 3 * h)
            {
                if ((w2 & 1) && (w > 2)) // prefer even splits of the long side
                {
                    ax2 += dax;
                    ay2 += day;
                }
                // split the long region in two
                if (in_region(x_dst, y_dst, x, y, ax2, ay2, bx, by))
                {
                    return gilbert_xy2d(x_dst, y_dst, x, y, ax2, ay2, bx, by);
                }
                const sfc_key_t base = static_cast<sfc_key_t>(iabs((ax2 + ay2) * (bx + by)));
                return base + gilbert_xy2d(x_dst, y_dst, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
            }

            if ((h2 & 1) && (h > 2))
            {
                bx2 += dbx;
                by2 += dby;
            }
            // split in three: up (along B2), across (along A), down (reversed)
            if (in_region(x_dst, y_dst, x, y, bx2, by2, ax2, ay2))
            {
                return gilbert_xy2d(x_dst, y_dst, x, y, bx2, by2, ax2, ay2);
            }
            sfc_key_t base = static_cast<sfc_key_t>(iabs((bx2 + by2) * (ax2 + ay2)));
            if (in_region(x_dst, y_dst, x + bx2, y + by2, ax, ay, bx - bx2, by - by2))
            {
                return base + gilbert_xy2d(x_dst, y_dst, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
            }
            base += static_cast<sfc_key_t>(iabs((ax + ay) * ((bx - bx2) + (by - by2))));
            return base
                 + gilbert_xy2d(x_dst, y_dst, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
        }

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
