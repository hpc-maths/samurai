// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The set algebra the call sites are written against. A backend is a struct
// of static functions over its own set expressions; set_call_sites.hpp
// reaches it through Algebra<Backend>, which also accepts level cell arrays
// wherever a set expression is expected. SamuraiStatic is samurai's static
// set algebra; set_algebra_dynamic_backend.hpp holds the dynamic one.
//
// A backend B provides, for every dim:
//
//   name                   a short name, the prefix of its benchmarks
//   lca_t<dim>             its level cell array: what self() reads, what
//                          materialize() and builders return
//   self(lca)              the set expression of the cells of lca
//   intersection(s...)     two operands or more
//   union_(s...)           one operand or more
//   difference(s...)       the cells of the first operand in none of the others
//   translate(s, t)        t: a samurai::DirectionVector<dim> or an xtensor
//                          expression of one
//   expand(s, w)           by w cells along every axis, corners included
//   expand(s, widths)      by widths[d] cells along axis d (std::array<int, dim>)
//   expand(s, w, mask)     by w cells along the axes d where mask[d]
//   contract(s, w)         by w cells along every axis (w: std::size_t)
//   contract(s, w, mask)   by w cells along the axes d where mask[d]
//   on(s, level)           the set projection of s to level
//   traverse(s, f)         calls f(interval, index) for each interval of s, in
//                          order: interval.start and interval.end bound the
//                          cells [start, end) along x, index[k] is the
//                          coordinate along axis k + 1
//   empty(s)               whether s has no cell
//   materialize(s)         the cells of s as an lca_t<dim>
//   builder<dim>(level)    an empty builder b of a level cell array:
//                          b.add_interval(interval, index), in any order, then
//                          b.build() returns the lca_t<dim>
//
// The level cell arrays a set expression reads outlive it.

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/stencil.hpp>
#include <samurai/subset/node.hpp>

// The layers of Algebra and SamuraiStatic only forward to samurai: always
// inlined, so that the static backend compiles to the calls of the library
// and does not change the inlining decisions around them.
#if defined(__GNUC__) || defined(__clang__)
#define BENCH_FORWARD [[gnu::always_inline]] inline
#else
#define BENCH_FORWARD inline
#endif

namespace bench
{
    // Checked on 2D sets, one use of each operation: a backend that fails it
    // lacks one of them or has another signature.
    template <class B>
    concept SetAlgebraBackend = requires(const typename B::template lca_t<2>& lca,
                                         const samurai::DirectionVector<2>& t,
                                         const std::array<int, 2>& widths,
                                         const std::array<bool, 2>& mask,
                                         std::size_t width,
                                         std::size_t level,
                                         const samurai::Interval<int>& interval,
                                         const samurai::LevelCellList<2>::index_yz_t& index) {
        { B::name } -> std::convertible_to<const char*>;
        B::intersection(B::self(lca), B::self(lca), B::self(lca));
        B::union_(B::self(lca));
        B::union_(B::self(lca), B::self(lca));
        B::difference(B::self(lca), B::self(lca));
        B::translate(B::self(lca), t);
        B::expand(B::self(lca), 1);
        B::expand(B::self(lca), widths);
        B::expand(B::self(lca), 1, mask);
        B::contract(B::self(lca), width);
        B::contract(B::self(lca), width, mask);
        B::on(B::self(lca), level);
        B::traverse(B::self(lca), [](const auto&, const auto&) {});
        { B::empty(B::self(lca)) } -> std::convertible_to<bool>;
        { B::materialize(B::self(lca)) } -> std::same_as<typename B::template lca_t<2>>;
        B::template builder<2>(level).add_interval(interval, index);
        { B::template builder<2>(level).build() } -> std::same_as<typename B::template lca_t<2>>;
    };

    // What the call sites call: the operations of backend B, whose operands
    // may also be level cell arrays of B, read through B::self as samurai's
    // own functions do.
    template <class B>
    struct Algebra
    {
        template <class S>
        BENCH_FORWARD static decltype(auto) lift(const S& s)
        {
            if constexpr (requires { B::self(s); })
            {
                return B::self(s);
            }
            else
            {
                return s;
            }
        }

        template <class... S>
        BENCH_FORWARD static auto intersection(const S&... s)
        {
            return B::intersection(lift(s)...);
        }

        template <class... S>
        BENCH_FORWARD static auto union_(const S&... s)
        {
            return B::union_(lift(s)...);
        }

        template <class... S>
        BENCH_FORWARD static auto difference(const S&... s)
        {
            return B::difference(lift(s)...);
        }

        template <class S, class T>
        BENCH_FORWARD static auto translate(const S& s, const T& t)
        {
            return B::translate(lift(s), t);
        }

        template <class S, class... Widths>
        BENCH_FORWARD static auto expand(const S& s, const Widths&... widths)
        {
            return B::expand(lift(s), widths...);
        }

        template <class S, class... Widths>
        BENCH_FORWARD static auto contract(const S& s, const Widths&... widths)
        {
            return B::contract(lift(s), widths...);
        }

        template <class S>
        BENCH_FORWARD static auto on(const S& s, std::size_t level)
        {
            return B::on(lift(s), level);
        }

        template <class S, class F>
        BENCH_FORWARD static void traverse(const S& s, F&& f)
        {
            B::traverse(lift(s), std::forward<F>(f));
        }

        template <class S>
        BENCH_FORWARD static bool empty(const S& s)
        {
            return B::empty(lift(s));
        }

        template <class S>
        BENCH_FORWARD static auto materialize(const S& s)
        {
            return B::materialize(lift(s));
        }

        template <std::size_t dim>
        BENCH_FORWARD static auto builder(std::size_t level)
        {
            return B::template builder<dim>(level);
        }
    };

    // Builds a samurai LevelCellArray through a LevelCellList, as the library
    // does.
    template <std::size_t dim>
    class SamuraiBuilder
    {
      public:

        explicit SamuraiBuilder(std::size_t level)
            : m_lcl(level)
        {
        }

        template <class Interval, class Index>
        void add_interval(const Interval& i, const Index& index)
        {
            m_lcl[index].add_interval({i.start, i.end});
        }

        samurai::LevelCellArray<dim> build() const
        {
            return samurai::LevelCellArray<dim>(m_lcl);
        }

      private:

        samurai::LevelCellList<dim> m_lcl;
    };

    // samurai's static set algebra (include/samurai/subset/), as the library
    // compiles it. The expansion by one width is nestedExpand, which the
    // library calls.
    struct SamuraiStatic
    {
        static constexpr const char* name = "static";

        template <std::size_t dim>
        using lca_t = samurai::LevelCellArray<dim>;

        template <std::size_t dim, class TInterval>
        BENCH_FORWARD static auto self(const samurai::LevelCellArray<dim, TInterval>& lca)
        {
            return samurai::self(lca);
        }

        template <class... S>
        BENCH_FORWARD static auto intersection(const S&... s)
        {
            return samurai::intersection(s...);
        }

        template <class... S>
        BENCH_FORWARD static auto union_(const S&... s)
        {
            return samurai::union_(s...);
        }

        template <class... S>
        BENCH_FORWARD static auto difference(const S&... s)
        {
            return samurai::difference(s...);
        }

        template <class S, class T>
        BENCH_FORWARD static auto translate(const S& s, const T& t)
        {
            return samurai::translate(s, t);
        }

        template <class S>
        BENCH_FORWARD static auto expand(const S& s, int width)
        {
            return samurai::nestedExpand(s, width);
        }

        template <class S>
        BENCH_FORWARD static auto expand(const S& s, const std::array<int, S::dim>& widths)
        {
            return samurai::expand(s, widths);
        }

        template <class S>
        BENCH_FORWARD static auto expand(const S& s, int width, const std::array<bool, S::dim>& mask)
        {
            return samurai::expand(s, width, mask);
        }

        template <class S>
        BENCH_FORWARD static auto contract(const S& s, std::size_t width)
        {
            return samurai::contract(s, width);
        }

        template <class S>
        BENCH_FORWARD static auto contract(const S& s, std::size_t width, const std::array<bool, S::dim>& mask)
        {
            return samurai::contract(s, width, mask);
        }

        template <class S>
        BENCH_FORWARD static auto on(const S& s, std::size_t level)
        {
            return s.on(level);
        }

        template <class S, class F>
        BENCH_FORWARD static void traverse(const S& s, F&& f)
        {
            s(std::forward<F>(f));
        }

        template <class S>
        BENCH_FORWARD static bool empty(const S& s)
        {
            return s.empty();
        }

        template <class S>
        BENCH_FORWARD static auto materialize(const S& s)
        {
            return s.to_lca();
        }

        template <std::size_t dim>
        static SamuraiBuilder<dim> builder(std::size_t level)
        {
            return SamuraiBuilder<dim>(level);
        }
    };

    static_assert(SetAlgebraBackend<SamuraiStatic>);
} // namespace bench
