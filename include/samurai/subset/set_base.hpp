// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>

#include <utility>

#include "traversers/set_traverser_base.hpp"

namespace samurai
{
    ////////////////////////////////////////////////////////////////////////
    //// Forward Declarations
    ////////////////////////////////////////////////////////////////////////

    template <class Set>
    struct SetTraits;

    template <class Derived>
    class SetBase;

    template <class Set>
    class Projection;

    template <class Set, class Func>
    void apply(const SetBase<Set>& set, Func&& func);

    ////////////////////////////////////////////////////////////////////////
    //// Class definition
    ////////////////////////////////////////////////////////////////////////

    template <class Derived>
    class SetBase
    {
        using DerivedTraits = SetTraits<Derived>;

      public:

        template <std::size_t d>
        using traverser_t = typename DerivedTraits::template traverser_t<d>;
        using interval_t  = typename traverser_t<0>::interval_t;
        using value_t     = typename interval_t::value_t;

        using to_lca_t       = LevelCellArray<DerivedTraits::dim, interval_t>;
        using to_lca_coord_t = typename to_lca_t::coords_t;

        static constexpr std::size_t dim = DerivedTraits::dim;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        inline std::size_t level() const
        {
            return derived_cast().level_impl();
        }

        inline bool exist() const
        {
            return derived_cast().exist_impl();
        }

        inline bool empty() const
        {
            return derived_cast().empty_impl();
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return derived_cast().get_traverser_impl(index, d_ic);
        }

        inline Projection<Derived> on(const std::size_t level);

        template <class Func>
        void operator()(Func&& func) const
        {
            apply(derived_cast(), std::forward<Func>(func));
        }

        template <class... ApplyOp>
        void apply_op(ApplyOp&&... op) const
        {
            const std::size_t l = level();

            auto func = [l, &op...](auto& interval, auto& index)
            {
                (op(l, interval, index), ...);
            };
            apply(derived_cast(), func);
        }

        to_lca_t to_lca() const
        {
            return to_lca_t(*this);
        }

        to_lca_t to_lca(const to_lca_coord_t& origin_point, const double scaling_factor) const
        {
            return to_lca_t(*this, origin_point, scaling_factor);
        }

      protected:

        inline bool empty_default_impl() const
        {
            xt::xtensor_fixed<int, xt::xshape<dim - 1>> index;
            return empty_default_impl_rec(index, std::integral_constant<std::size_t, dim - 1>{});
        }

        template <class index_t, std::size_t d>
        bool empty_default_impl_rec(index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            using current_interval_t = typename traverser_t<d>::current_interval_t;

            for (traverser_t<d> traverser = get_traverser(index, d_ic); !traverser.is_empty(); traverser.next_interval())
            {
                current_interval_t interval = traverser.current_interval();

                if constexpr (d == 0)
                {
                    return false;
                }
                else
                {
                    for (index[d - 1] = interval.start; index[d - 1] != interval.end; ++index[d - 1])
                    {
                        if (not empty_default_impl_rec(index, std::integral_constant<std::size_t, d - 1>{}))
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    };

#define SAMURAI_SET_TYPEDEFS                                    \
    using Base = SetBase<Self>;                                 \
                                                                \
    template <std::size_t d>                                    \
    using traverser_t = typename Base::template traverser_t<d>; \
                                                                \
    using interval_t = typename Base::interval_t;               \
    using value_t    = typename Base::value_t;

#define SAMURAI_SET_CONSTEXPRS static constexpr std::size_t dim = Base::dim;

    template <typename T>
    struct IsSet : std::bool_constant<std::is_base_of<SetBase<T>, T>::value>
    {
    };

    template <class Set>
    const Set& self(const SetBase<Set>& set)
    {
        return set.derived_cast();
    }

} // namespace samurai

#include "projection.hpp"

namespace samurai
{

    template <class Derived>
    Projection<Derived> SetBase<Derived>::on(const std::size_t level)
    {
        return Projection<Derived>(derived_cast(), level);
    }

}
