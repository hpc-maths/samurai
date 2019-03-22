#pragma once

#include <iostream>
#include <tuple>
#include <utility>

namespace mure
{
    namespace detail
    {
        template <std::size_t I>
        struct getter
        {
            template <class Arg, class... Args>
            static constexpr decltype(auto) get(Arg&& /*arg*/, Args&&... args) noexcept
            {
                return getter<I - 1>::get(std::forward<Args>(args)...);
            }
        };

        template <>
        struct getter<0>
        {
            template <class Arg, class... Args>
            static constexpr Arg&& get(Arg&& arg, Args&&... /*args*/) noexcept
            {
                return std::forward<Arg>(arg);
            }
        };
    }

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept
    {
        static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
        return detail::getter<I>::get(std::forward<Args>(args)...);
    }

    template <std::size_t I>
    struct picker
    {
        template <class... Args>
        constexpr decltype(auto) operator()(std::size_t dim, Args&&... args) const
        {
            return argument<I>(std::forward<Args>(args)...);
        }
    };

    picker<0u> _1;
    picker<1u> _2;
    picker<2u> _3;
    picker<3u> _4;
    picker<4u> _5;
    picker<5u> _6;
    picker<6u> _7;
    picker<7u> _8;
    picker<8u> _9;
    picker<9u> _10;

    template <class Func, class... F>
    struct func_node
    {
        using tuple_type = std::tuple<F...>;
        using functor_type = Func;

        template <class FuncA, class... FA>
        func_node(FuncA&& f, FA&&... fa)
            : m_op(std::forward<FA>(fa)...), m_f(std::forward<FuncA>(f))
        {
        }

        template <class... Args>
        decltype(auto) operator()(std::size_t dim, Args&&... args) const
        {
            return access_impl(std::make_index_sequence<sizeof...(F)>(), dim, std::forward<Args>(args)...);
        }

        template <std::size_t... I, class... Args>
        decltype(auto) access_impl(std::index_sequence<I...>, std::size_t dim, Args&&... args) const
        {
            return m_f(dim, std::get<I>(m_op)(dim, std::forward<Args>(args)...)...);
        }

        tuple_type m_op;
        functor_type m_f;
    };

    template <class Func, class... F>
    auto make_func_node(Func&& func, F&&... f)
    {
        using func_type = func_node<Func, F...>;
        return func_type(std::forward<Func>(func), std::forward<F>(f)...);
    }
}