// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "utils.hpp"

namespace samurai
{
    template <class value_t, std::size_t size, bool SOA = false>
    struct xtensor_container
    {
        using container_t = xt::xtensor<value_t, (size == 1) ? 1 : 2>;

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
    auto view(xtensor_container<value_t, 1, SOA>& container, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(xtensor_container<value_t, size, true>& container, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::all(), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(xtensor_container<value_t, size, false>& container, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, true>& container, const range_t<std::size_t>& range_item, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::range(range_item.start, range_item.end), xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, false>& container, const range_t<std::size_t>& range_item, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end), xt::range(range_item.start, range_item.end));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, true>& container, std::size_t item, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), item, xt::range(range.start, range.end));
    }

    template <class value_t, std::size_t size>
    auto view(xtensor_container<value_t, size, false>& container, std::size_t item, const range_t<std::size_t>& range)
    {
        return xt::view(container.data(), xt::range(range.start, range.end), item);
    }
}
