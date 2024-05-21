// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/xnoalias.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "utils.hpp"

namespace samurai
{
    template <class value_t, std::size_t size, bool SOA = false>
    struct xtensor_container
    {
        using container_t = xt::xtensor<value_t, (size == 1) ? 1 : 2>;

        xtensor_container() = default;

        xtensor_container(std::size_t dynamic_size)
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
            if constexpr (size == 1)
            {
                m_data.resize({dynamic_size});
            }
            else
            {
                if constexpr (SOA)
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

    template <class value_t, bool SOA>
    auto view(xtensor_container<value_t, 1, SOA>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(xtensor_container<value_t, size, true>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::all(), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(xtensor_container<value_t, size, false>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, true>& container, const range_t<long long>& range_item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range_item.start, range_item.end, range_item.step), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, false>& container, const range_t<long long>& range_item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end, range.step), xt::range(range_item.start, range_item.end, range_item.step));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, true>& container, std::size_t item, const range_t<long long>& range)
    {
        return xt::view(container.data(), item, xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, false>& container, std::size_t item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end, range.step), item);
    }

    template <class value_t, bool SOA>
    auto view(const xtensor_container<value_t, 1, SOA>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(const xtensor_container<value_t, size, true>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::all(), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(const xtensor_container<value_t, size, false>& container, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, true>& container, const range_t<long long>& range_item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range_item.start, range_item.end, range_item.step), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, false>& container, const range_t<long long>& range_item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end), xt::range(range_item.start, range_item.end, range_item.step));
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, true>& container, std::size_t item, const range_t<long long>& range)
    {
        return xt::view(container.data(), item, xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, false>& container, std::size_t item, const range_t<long long>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end), item);
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, false>& container, std::size_t index)
    {
        return xt::view(container.data(), index);
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, false>& container, std::size_t index)
    {
        return xt::view(container.data(), index);
    }

    template <class value_t, std::size_t size>
    auto view(const xtensor_container<value_t, size, true>& container, std::size_t index)
    {
        return xt::view(container.data(), xt::all(), index);
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, true>& container, std::size_t index)
    {
        return xt::view(container.data(), xt::all(), index);
    }

    template <class D>
    auto eval(const xt::xexpression<D>& exp)
    {
        return xt::eval(exp);
    }

    template <class D>
    auto shape(const xt::xexpression<D>& exp, std::size_t axis)
    {
        return exp.derived_cast().shape(axis);
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

    template <class D>
    auto abs(const xt::xcontainer<D>& exp)
    {
        return xt::abs(exp);
    }

    template <class D>
    auto sum(const xt::xcontainer<D>& exp)
    {
        return xt::sum(exp);
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
}
