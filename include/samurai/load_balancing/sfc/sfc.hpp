// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstdint>
#include <string>

namespace samurai::load_balancing
{
    /// 1D index along a space-filling curve. 64 bits bound the usable
    /// coordinate range per dimension (see `SFCCurve::max_bits`).
    using sfc_key_t = std::uint64_t;

    /**
     * CRTP interface of a space-filling curve.
     *
     * A curve maps non-negative logical coordinates (one per dimension,
     * normalized to a common refinement level by the caller) to a 1D key
     * preserving spatial locality: cells close on the curve are close in
     * space. The curve is the locality engine of the SFC partitioning
     * strategy (strategies/sfc.hpp).
     *
     * Contract for a flavor F : SFCCurve<F>:
     *  - `sfc_key_t key_2d(coords)` and `sfc_key_t key_3d(coords)` where
     *    `coords(d)` is an unsigned 32-bit coordinate with at most
     *    `max_bits(dim)` significant bits;
     *  - `std::string name()`.
     * The inverse mapping (key -> coordinates) is NOT part of the contract:
     * partitioning only needs the forward direction. Morton provides one as a
     * flavor-specific extra (useful for debugging and testing).
     */
    template <class Flavor>
    class SFCCurve
    {
      public:

        /// Usable bits per coordinate so that the key fits in 64 bits:
        /// 32 in 1D/2D, 21 in 3D (=> deepest usable level: 21 in 3D).
        static constexpr unsigned max_bits(std::size_t dim)
        {
            return dim <= 2 ? 32U : 64U / static_cast<unsigned>(dim);
        }

        /// 1D key of the point `p` (p(d) >= 0, fitting in max_bits(dim) bits).
        /// Maps the full 2^max_bits square/cube grid.
        template <std::size_t dim, class Coord>
        sfc_key_t key(const Coord& p) const
        {
            static_assert(1 <= dim && dim <= 3, "space-filling curves are implemented for dim 1, 2, 3");
            if constexpr (dim == 1)
            {
                return static_cast<sfc_key_t>(p(0)); // the identity curve
            }
            else if constexpr (dim == 2)
            {
                return derived().key_2d(p);
            }
            else
            {
                return derived().key_3d(p);
            }
        }

        /// 1D key of `p` inside the `n(0) x n(1) [x n(2)]` bounding box. A flavor
        /// may exploit the box extent to preserve locality on non-square domains
        /// (Hilbert lays a generalized curve); flavors that don't simply ignore
        /// `n` and fall back to the square mapping. `p(d)` must satisfy
        /// 0 <= p(d) < n(d).
        template <std::size_t dim, class Coord, class Extent>
        sfc_key_t key(const Coord& p, const Extent& n) const
        {
            static_assert(1 <= dim && dim <= 3, "space-filling curves are implemented for dim 1, 2, 3");
            if constexpr (dim == 1)
            {
                return static_cast<sfc_key_t>(p(0)); // the identity curve
            }
            else if constexpr (dim == 2)
            {
                return derived().key_2d(p, n);
            }
            else
            {
                return derived().key_3d(p, n);
            }
        }

        std::string name() const
        {
            return derived().name_impl();
        }

      private:

        const Flavor& derived() const
        {
            return static_cast<const Flavor&>(*this);
        }
    };
}
