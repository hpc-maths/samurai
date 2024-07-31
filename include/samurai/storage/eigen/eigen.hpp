// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <Eigen/Core>

#include "../utils.hpp"

namespace samurai
{
    namespace placeholders
    {
        static constexpr Eigen::internal::all_t _;

    }

    namespace detail
    {
        template <class value_t, std::size_t size, bool SOA, class = void>
        struct eigen_type;

        template <class value_t, bool SOA>
        struct eigen_type<value_t, 1, SOA>
        {
            using type = Eigen::Array<value_t, Eigen::Dynamic, 1>;
        };

        template <class value_t, std::size_t size>
        struct eigen_type<value_t, size, true, std::enable_if_t<(size > 1)>>
        {
            using type = Eigen::Array<value_t, size, Eigen::Dynamic, Eigen::RowMajor>;
        };

        template <class value_t, std::size_t size>
        struct eigen_type<value_t, size, false, std::enable_if_t<(size > 1)>>
        {
            using type = Eigen::Array<value_t, Eigen::Dynamic, size, Eigen::ColMajor>;
        };

        template <class value_t, std::size_t size, bool SOA>
        using eigen_type_t = typename eigen_type<value_t, size, SOA>::type;

    }

    template <class value_t, std::size_t size, bool SOA>
    struct eigen_container
    {
        using container_t = detail::eigen_type_t<value_t, size, SOA>;
        using size_type   = Eigen::Index;

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

        void resize(size_type dynamic_size)
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
        return container.data()(Eigen::placeholders::all, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), Eigen::placeholders::all);
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
        return container.data()(Eigen::placeholders::all, Eigen::seq(range.start, range.end - 1, range.step));
    }

    template <class value_t, std::size_t size, typename = std::enable_if_t<(size > 1)>>
    auto view(const eigen_container<value_t, size, false>& container, const range_t<Eigen::Index>& range)
    {
        return container.data()(Eigen::seq(range.start, range.end - 1, range.step), Eigen::placeholders::all);
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

    template <class value_t, std::size_t size>
    auto view(const eigen_container<value_t, size, false>& container, Eigen::Index index)
    {
        return container.data()(index, Eigen::placeholders::all);
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, false>& container, Eigen::Index index)
    {
        return container.data()(index, Eigen::placeholders::all);
    }

    template <class value_t, std::size_t size>
    auto view(const eigen_container<value_t, size, true>& container, Eigen::Index index)
    {
        return container.data()(Eigen::placeholders::all, index);
    }

    template <class value_t, std::size_t size>
    auto view(eigen_container<value_t, size, true>& container, Eigen::Index index)
    {
        return container.data()(Eigen::placeholders::all, index);
    }

    template <class D>
    auto eval(const Eigen::EigenBase<D>& exp)
    {
        return exp.derived().eval();
    }

    template <class D>
    auto shape(const Eigen::EigenBase<D>& exp, std::size_t axis)
    {
        assert(axis < 2);
        if (axis == 0)
        {
            return exp.derived().rows();
        }
        return exp.derived().cols();
    }

    template <class D>
    auto noalias(const Eigen::EigenBase<D>& exp)
    {
        return exp.derived();
    }

    template <class T1, class T2>
    auto range(const T1& start, const T2& end)
    {
        return Eigen::seq(start, end - 1);
    }

    template <class T>
    auto range(const T& start)
    {
        return Eigen::seq(start, Eigen::placeholders::all);
    }

    template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options, class Range>
    auto view(Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& container, const Range& range)
    {
        if constexpr (ColsAtCompileTime > 1)
        {
            return container(range, Eigen::placeholders::all);
        }
        else
        {
            return container(range);
        }
    }

    template <class D, class Range>
    auto view(Eigen::EigenBase<D>& exp, const Range& range)
    {
        if constexpr (D::ColsAtCompileTime > 1)
        {
            return exp.derived()(range, Eigen::placeholders::all);
        }
        else
        {
            return exp.derived()(range);
        }
    }

    namespace math
    {
        template <class D>
        auto abs(const Eigen::EigenBase<D>& exp)
        {
            return exp.derived().cwiseAbs();
        }

        template <class D>
        auto sum(const Eigen::EigenBase<D>& exp)
        {
            return exp.derived().sum();
        }

        template <std::size_t axis, class D>
        auto sum(const Eigen::EigenBase<D>& exp)
        {
            static_assert(axis < 2);
            if constexpr (axis == 0)
            {
                return exp.derived().rowwise().sum();
            }
            else
            {
                return exp.derived().colwise().sum();
            }
        }

        template <class D1, class D2>
        auto minimum(const Eigen::EigenBase<D1>& exp1, const Eigen::EigenBase<D2>& exp2)
        {
            return exp1.derived().cwiseMin(exp2.derived());
        }

        template <class D>
        auto minimum(double s, const Eigen::EigenBase<D>& exp)
        {
            return exp.derived().cwiseMin(s);
        }

        template <class Scalar, class D>
        auto minimum(const Eigen::EigenBase<D>& exp, Scalar s)
        {
            return minimum(s, exp);
        }

        template <class D1, class D2>
        auto maximum(const Eigen::EigenBase<D1>& exp1, const Eigen::EigenBase<D2>& exp2)
        {
            return exp1.derived().cwiseMax(exp2.derived());
        }

        template <class Scalar, class D>
        auto maximum(Scalar s, const Eigen::ArrayBase<D>& exp)
        {
            return exp.derived().cwiseMax(s);
        }

        template <class Scalar, class D>
        auto maximum(const Eigen::EigenBase<D>& exp, Scalar s)
        {
            return maximum(s, exp);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto minimum(const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp1,
                     const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp2)
        {
            return exp1.cwiseMin(exp2);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto minimum(double s, const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp)
        {
            return exp.cwiseMin(s);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto minimum(const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp, Scalar s)
        {
            return minimum(s, exp);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto maximum(const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp1,
                     const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp2)
        {
            return exp1.cwiseMax(exp2);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto maximum(Scalar s, const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp)
        {
            return exp.cwiseMax(s);
        }

        template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
        auto maximum(const Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& exp, Scalar s)
        {
            return maximum(s, exp);
        }

        using Eigen::abs;
        using Eigen::exp;
        using Eigen::pow;
        using Eigen::tanh;

        template <class Scalar>
        auto arange(const Scalar& low, const Scalar& high)
        {
            return Eigen::Array<Scalar, Eigen::Dynamic, 1>::LinSpaced(static_cast<Eigen::Index>(high - low + 1), low, high);
        }
    }

    template <class D>
    auto operator>(const Eigen::EigenBase<D>& exp, double x)
    {
        return exp.derived().array() > x;
    }

    template <class D>
    auto operator<(const Eigen::EigenBase<D>& exp, double x)
    {
        return exp.derived().array() < x;
    }

    template <class D>
    auto operator>=(const Eigen::EigenBase<D>& exp, double x)
    {
        return exp.derived().array() >= x;
    }

    template <class DST, class CRIT, class FUNC>
    void apply_on_masked(Eigen::EigenBase<DST>& dst, const Eigen::EigenBase<CRIT>& criteria, FUNC&& func)
    {
        for (Eigen::Index i = 0; i < criteria.size(); ++i)
        {
            if (criteria.derived()(i))
            {
                func(dst.derived()(i));
            }
        }
    }

    template <class DST, class CRIT, class FUNC>
    void apply_on_masked(Eigen::EigenBase<DST>&& dst, const Eigen::EigenBase<CRIT>& criteria, FUNC&& func)
    {
        for (Eigen::Index i = 0; i < criteria.size(); ++i)
        {
            if (criteria.derived()(i))
            {
                func(dst.derived()(i));
            }
        }
    }

    template <class CRIT, class FUNC>
    void apply_on_masked(const Eigen::EigenBase<CRIT>& criteria, FUNC&& func)
    {
        for (Eigen::Index i = 0; i < criteria.size(); ++i)
        {
            if (criteria.derived()(i))
            {
                func(i);
            }
        }
    }

    template <class D>
    auto zeros_like(const Eigen::EigenBase<D>& container)
    {
        using Base = typename D::PlainArray;
        Base res   = Base::Zero(container.rows(), container.cols());
        return res;
    }
}
