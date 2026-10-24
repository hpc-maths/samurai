// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// samurai's dynamic (runtime) set algebra (include/samurai/subset/dynamic/)
// as a backend of the call sites: every operation allocates a node, as an
// expression built at runtime does. See set_algebra_backend.hpp for the
// interface.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <samurai/stencil.hpp>
#include <samurai/subset/dynamic/dynamic.hpp>

#include "set_algebra_backend.hpp"

namespace bench
{
    struct SamuraiDynamic
    {
        static constexpr const char* name = "dynamic";

        template <std::size_t dim>
        using lca_t = samurai::LevelCellArray<dim>;

        template <std::size_t dim, class TInterval>
        using set_t = samurai::DynamicSet<dim, TInterval>;

        template <std::size_t dim, class TInterval>
        static auto self(const samurai::LevelCellArray<dim, TInterval>& lca)
        {
            return samurai::dyn::self(lca);
        }

        template <class... S>
        static auto intersection(const S&... s)
        {
            return samurai::dyn::intersection(s...);
        }

        template <class... S>
        static auto union_(const S&... s)
        {
            return samurai::dyn::union_(s...);
        }

        template <class... S>
        static auto difference(const S&... s)
        {
            return samurai::dyn::difference(s...);
        }

        // The node keeps its translation: an xtensor expression is evaluated
        // here, before its operands go away.
        template <std::size_t dim, class TInterval>
        static auto translate(const set_t<dim, TInterval>& s, const std::type_identity_t<samurai::DirectionVector<dim>>& t)
        {
            return samurai::dyn::translate(s, t);
        }

        template <std::size_t dim, class TInterval>
        static auto expand(const set_t<dim, TInterval>& s, int width)
        {
            return samurai::dyn::expand(s, width);
        }

        // samurai::dyn has no expansion per axis: the node wraps the static one.
        template <std::size_t dim, class TInterval>
        static auto expand(const set_t<dim, TInterval>& s, const std::array<int, dim>& widths)
        {
            return samurai::make_node<dim, TInterval>({s.ptr()},
                                                      [widths](const auto& children)
                                                      {
                                                          return samurai::expand(samurai::as_static_set(children[0]), widths);
                                                      });
        }

        template <std::size_t dim, class TInterval>
        static auto expand(const set_t<dim, TInterval>& s, int width, const std::array<bool, dim>& mask)
        {
            return samurai::make_node<dim, TInterval>({s.ptr()},
                                                      [width, mask](const auto& children)
                                                      {
                                                          return samurai::expand(samurai::as_static_set(children[0]), width, mask);
                                                      });
        }

        template <std::size_t dim, class TInterval>
        static auto contract(const set_t<dim, TInterval>& s, std::size_t width)
        {
            return samurai::dyn::contract(s, width);
        }

        template <std::size_t dim, class TInterval>
        static auto contract(const set_t<dim, TInterval>& s, std::size_t width, const std::array<bool, dim>& mask)
        {
            return samurai::dyn::contract(s, width, mask);
        }

        template <std::size_t dim, class TInterval>
        static auto on(const set_t<dim, TInterval>& s, std::size_t level)
        {
            return s.on(level);
        }

        template <std::size_t dim, class TInterval, class F>
        static void traverse(const set_t<dim, TInterval>& s, F&& f)
        {
            s(std::forward<F>(f));
        }

        template <std::size_t dim, class TInterval>
        static bool empty(const set_t<dim, TInterval>& s)
        {
            return s.ptr()->empty();
        }

        template <std::size_t dim, class TInterval>
        static auto materialize(const set_t<dim, TInterval>& s)
        {
            return s.to_lca();
        }

        template <std::size_t dim>
        static SamuraiBuilder<dim> builder(std::size_t level)
        {
            return SamuraiBuilder<dim>(level);
        }
    };

    static_assert(SetAlgebraBackend<SamuraiDynamic>);
} // namespace bench
