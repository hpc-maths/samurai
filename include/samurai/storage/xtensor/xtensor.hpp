// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// #include <xtensor/xlayout.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/views/xview.hpp>

#include "../utils.hpp"

namespace samurai
{
    namespace placeholders
    {
        using xt::all;
        using xt::placeholders::_;
    }

    /**
     * The values of a field: one per cell for a scalar field, an array of shape {cells, size} for a
     * vector field, whatever its number of components.
     */
    template <class value_t, std::size_t size, bool is_scalar>
    struct xtensor_container
    {
        static_assert(!is_scalar || size == 1, "a scalar field has one component");

        using container_t = xt::xtensor<value_t, is_scalar ? 1 : 2, xt::layout_type::row_major>;
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
            if constexpr (is_scalar)
            {
                m_data.resize({dynamic_size});
            }
            else
            {
                m_data.resize({dynamic_size, size});
            }
        }

      private:

        container_t m_data;
    };

    namespace detail
    {
        template <class T>
        struct xtensor_container_traits;

        template <class value_t, std::size_t size_, bool is_scalar_>
        struct xtensor_container_traits<xtensor_container<value_t, size_, is_scalar_>>
        {
            static constexpr std::size_t size = size_;
            static constexpr bool is_scalar   = is_scalar_;
        };
    }

    /**
     * An xtensor_container, const or not: the views below keep its constness.
     */
    template <class T>
    concept xtensor_container_like = requires { detail::xtensor_container_traits<std::remove_const_t<T>>::size; };

    /**
     * The xtensor_container of a scalar field: it has no component axis.
     */
    template <class T>
    concept xtensor_scalar_container = xtensor_container_like<T> && detail::xtensor_container_traits<std::remove_const_t<T>>::is_scalar;

    //////////
    // VIEW //
    //////////

    template <xtensor_container_like Container>
    auto view(Container& container, const range_t<long long>& range)
    {
        using traits = detail::xtensor_container_traits<std::remove_const_t<Container>>;
        return xt::view(container.data(), xt::range(range.start, range.end, range.step));
    }

    template <xtensor_container_like Container>
    auto view(Container& container, const range_t<std::size_t>& range_item, const range_t<long long>& range)
    {
        static_assert(!xtensor_scalar_container<Container>, "a scalar field has no component axis");
        return xt::view(container.data(),
                        xt::range(range.start, range.end, range.step),
                        xt::range(range_item.start, range_item.end, range_item.step));
    }

    template <xtensor_container_like Container>
    auto view(Container& container, std::size_t item, const range_t<long long>& range)
    {
        static_assert(!xtensor_scalar_container<Container>, "a scalar field has no component axis");
        return xt::view(container.data(), xt::range(range.start, range.end, range.step), item);
    }

    template <xtensor_container_like Container>
    auto view(Container& container, std::size_t index)
    {
        using traits = detail::xtensor_container_traits<std::remove_const_t<Container>>;
        return xt::view(container.data(), index);
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
