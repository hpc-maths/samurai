// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <Eigen/Core>

#include "utils.hpp"

namespace samurai
{
    namespace detail
    {
        template <class value_t, std::size_t size, bool SOA, class = void>
        struct eigen_type;

        template <class value_t, bool SOA>
        struct eigen_type<value_t, 1, SOA>
        {
            using type = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
        };

        template <class value_t, std::size_t size>
        struct eigen_type<value_t, size, true, std::enable_if_t<(size > 1)>>
        {
            using type = Eigen::Matrix<value_t, size, Eigen::Dynamic, Eigen::RowMajor>;
        };

        template <class value_t, std::size_t size>
        struct eigen_type<value_t, size, false, std::enable_if_t<(size > 1)>>
        {
            using type = Eigen::Matrix<value_t, Eigen::Dynamic, size, Eigen::RowMajor>;
        };

        template <class value_t, std::size_t size, bool SOA>
        using eigen_type_t = typename eigen_type<value_t, size, SOA>::type;

    }

    template <class value_t, std::size_t size, bool SOA>
    struct eigen_container
    {
        using container_t = detail::eigen_type_t<value_t, size, SOA>;

        eigen_container() = default;

        eigen_container(std::size_t dynamic_size)
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
                m_data.resize(dynamic_size);
            }
            else
            {
                if constexpr (SOA)
                {
                    m_data.resize(size, dynamic_size);
                }
                else
                {
                    m_data.resize(dynamic_size, size);
                }
            }
        }

      private:

        container_t m_data;
    };

    template <class value_t, bool SOA>
    auto view(eigen_container<value_t, 1, SOA>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(eigen_container<value_t, size, true>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::all, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), Eigen::all);
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, true>& container, const range_t<Eigen::Index>& range_item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range_item.start, range_item.end - 1, range_item.step),
                                Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range_item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step),
                                Eigen::seq(range_item.start, range_item.end - 1, range_item.step));
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, true>& container, std::size_t item, const range_t<Eigen::Index>& range)
    {
        return container.data()(item, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, false>& container, std::size_t item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), item);
    }

    template <class value_t, bool SOA>
    auto view(const eigen_container<value_t, 1, SOA>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(const eigen_container<value_t, size, true>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::all, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(const eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), Eigen::all);
    }

    template <class value_t, std::size_t size>
    auto
    view(const eigen_container<value_t, size, true>& container, const range_t<Eigen::Index>& range_item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range_item.start, range_item.end - 1, range_item.step),
                                Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size>
    auto
    view(const eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range_item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step),
                                Eigen::seq(range_item.start, range_item.end - 1, range_item.step));
    }

    template <class value_t, std::size_t size>
    auto view(const eigen_container<value_t, size, true>& container, std::size_t item, const range_t<Eigen::Index>& range)
    {
        return container.data()(item, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size>
    auto view(const eigen_container<value_t, size, false>& container, std::size_t item, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), item);
    }

}
