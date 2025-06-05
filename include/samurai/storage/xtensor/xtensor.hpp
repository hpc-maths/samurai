// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// #include <xtensor/xlayout.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "../utils.hpp"

namespace samurai
{
    namespace placeholders
    {
        using xt::all;
        using xt::placeholders::_;
    }

    namespace detail
    {
        template <::samurai::layout_type L>
        struct xtensor_layout;

        template <>
        struct xtensor_layout<::samurai::layout_type::row_major>
        {
            static constexpr ::xt::layout_type value = ::xt::layout_type::row_major;
        };

        template <>
        struct xtensor_layout<::samurai::layout_type::column_major>
        {
            static constexpr ::xt::layout_type value = ::xt::layout_type::column_major;
        };

        template <layout_type L>
        static constexpr ::xt::layout_type xtensor_layout_v = xtensor_layout<L>::value;
    }

    template <class value_t, std::size_t size, bool SOA = false, bool can_collapse = true>
    struct xtensor_container
    {
        static constexpr layout_type static_layout = SAMURAI_DEFAULT_LAYOUT;
        using container_t = xt::xtensor<value_t, ((size == 1) && can_collapse) ? 1 : 2, detail::xtensor_layout_v<static_layout>>;
        using size_type   = std::size_t;

        xtensor_container() = default;

        explicit xtensor_container(std::size_t dynamic_size)
            : m_data()
        {
            resize(dynamic_size);
        }

        const container_t& data() const
        {
            return m_data;
        }

        container_t& data()
        {
            return m_data;
        }

        void resize(std::size_t dynamic_size)
        {
            if constexpr ((size == 1) && can_collapse)
            {
                m_data.resize({dynamic_size});
            }
            else
            {
                if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
                {
                    m_data.resize({size, dynamic_size});
                }
                else
                {
                    m_data.resize({dynamic_size, size});
                }
            }
        }

      private:

        container_t m_data;
    };

    ////////////////////////////////
    // VIEW: CONST IMPLEMENTATION //
    ////////////////////////////////

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const xtensor_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), xt::all(), xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(), xt::range(range.start, range.end, range.step));
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const xtensor_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(),
                            xt::range(range_item.start, range_item.end, range_item.step),
                            xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(),
                            xt::range(range.start, range.end, range.step),
                            xt::range(range_item.start, range_item.end, range_item.step));
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const xtensor_container<value_t, size, SOA, can_collapse>& container, std::size_t item, const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), item, xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(), xt::range(range.start, range.end, range.step), item);
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const xtensor_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;

        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), xt::all(), index);
        }
        else
        {
            return xt::view(container.data(), index);
        }
    }

    ////////////////////////////////////
    // VIEW: NON CONST IMPLEMENTATION //
    ////////////////////////////////////

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(xtensor_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), xt::all(), xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(), xt::range(range.start, range.end, range.step));
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(xtensor_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(),
                            xt::range(range_item.start, range_item.end, range_item.step),
                            xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(),
                            xt::range(range.start, range.end, range.step),
                            xt::range(range_item.start, range_item.end, range_item.step));
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(xtensor_container<value_t, size, SOA, can_collapse>& container, std::size_t item, const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), item, xt::range(range.start, range.end, range.step));
        }
        else
        {
            return xt::view(container.data(), xt::range(range.start, range.end, range.step), item);
        }
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(xtensor_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        static constexpr layout_type static_layout = xtensor_container<value_t, size, SOA, can_collapse>::static_layout;
        if constexpr (detail::static_size_first_v<size, SOA, can_collapse, static_layout>)
        {
            return xt::view(container.data(), xt::all(), index);
        }
        else
        {
            return xt::view(container.data(), index);
        }
    }

    template <class D>
    auto eval(const xt::xexpression<D>& exp)
    {
        return xt::eval(exp.derived_cast());
    }

    template <class D1, class D2>
    bool compare(const xt::xexpression<D1>& exp1, const xt::xexpression<D2>& exp2)
    {
        return exp1 == exp2;
    }

    template <class D>
    auto shape(const xt::xexpression<D>& exp, std::size_t axis)
    {
        return exp.derived_cast().shape(axis);
    }

    template <class D>
    auto shape(const xt::xexpression<D>& exp)
    {
        return exp.derived_cast().shape();
    }

    template <class D>
    auto noalias(const xt::xexpression<D>& exp)
    {
        return xt::noalias(exp);
    }

    template <class T1, class T2>
    auto range(const T1& start, const T2& end)
    {
        return xt::range(start, end);
    }

    template <class T>
    auto range(const T& start)
    {
        using namespace xt::placeholders;
        return xt::range(start, _);
    }

    template <class D, class Range>
    auto view(const xt::xcontainer<D>& container, const Range& range)
    {
        return xt::view(container, range);
    }

    using xt::view;

    template <class T>
    auto zeros(std::size_t size)
    {
        return xt::zeros<T>({size});
    }

    namespace math
    {
        using namespace xt::math;
        using xt::arange;
        using xt::maximum;
        using xt::minimum;
        using xt::transpose;

        template <class D>
        auto sum(xt::xexpression<D>&& exp)
        {
            return xt::sum(exp.derived_cast())[0];
        }

        template <class F, class... CT>
        auto sum(xt::xfunction<F, CT...>&& exp)
        {
            return xt::sum(exp)[0];
        }

        template <std::size_t axis, class D>
        auto sum(xt::xexpression<D>&& exp)
        {
            return xt::sum(exp.derived_cast(), {axis});
        }

        template <std::size_t axis, std::size_t size, class D>
        auto all_true(xt::xexpression<D>&& exp)
        {
            return xt::sum(exp.derived_cast(), {axis}) > (size - 1);
        }
    }

    template <class D>
    auto operator>(const xt::xcontainer<D>& exp, double x)
    {
        return exp > x;
    }

    template <class D>
    auto operator<(const xt::xcontainer<D>& exp, double x)
    {
        return exp < x;
    }

    template <class D>
    auto operator>(xt::xexpression<D>&& exp, double x)
    {
        return exp.derived_cast() > x;
    }

    template <class D>
    auto operator<(xt::xexpression<D>&& exp, double x)
    {
        return exp.derived_cast() < x;
    }

    template <class DST, class CRIT, class FUNC>
    void apply_on_masked(xt::xexpression<DST>& dst, const xt::xexpression<CRIT>& criteria, FUNC&& func)
    {
        for (std::size_t i = 0; i < criteria.derived_cast().size(); ++i)
        {
            if (criteria.derived_cast()(i))
            {
                func(dst.derived_cast()(i));
            }
        }
    }

    template <class DST, class CRIT, class FUNC>
    void apply_on_masked(xt::xexpression<DST>&& dst, const xt::xexpression<CRIT>& criteria, FUNC&& func)
    {
        for (std::size_t i = 0; i < criteria.derived_cast().size(); ++i)
        {
            if (criteria.derived_cast()(i))
            {
                func(dst.derived_cast()(i));
            }
        }
    }

    template <class CRIT, class FUNC>
    void apply_on_masked(const xt::xexpression<CRIT>& criteria, FUNC&& func)
    {
        for (std::size_t i = 0; i < criteria.derived_cast().size(); ++i)
        {
            if (criteria.derived_cast()(i))
            {
                func(i);
            }
        }
    }

    template <class D>
    auto zeros_like(xt::xexpression<D>&& exp)
    {
        return xt::zeros_like(exp.derived_cast());
    }
}
