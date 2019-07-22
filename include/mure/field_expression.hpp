#pragma once

#include "xtl/xtype_traits.hpp"

#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"

#include "cell.hpp"
#include "interval.hpp"

namespace mure
{
    struct field_expression_tag
    {
    };

    template<class E>
    struct is_field_expression
        : std::is_same<xt::xexpression_tag_t<E>, field_expression_tag>
    {
    };

    template<class... E>
    struct field_comparable : xtl::conjunction<is_field_expression<E>...>
    {
    };

    template<class D>
    class field_expression : public xt::xexpression<D> {
      public:
        using expression_tag = field_expression_tag;
    };

    template<class F, class R, class... CT>
    class field_function
        : public field_expression<field_function<F, R, CT...>> {
      public:
        using self_type = field_function<F, R, CT...>;
        using functor_type = std::remove_reference_t<F>;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using interval_t = Interval<int>;

        using expression_tag = field_expression_tag;

        template<
            class Func, class... CTA,
            class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        field_function(Func &&f, CTA &&... e) noexcept;

        template<class... T>
        auto const operator()(interval_t interval, T... index) const
        {
            return evaluate(std::make_index_sequence<sizeof...(CT)>(), interval,
                            index...);
        }

        template<class coord_index_t, std::size_t dim>
        auto const operator()(Cell<coord_index_t, dim> cell) const
        {
            return evaluate(std::make_index_sequence<sizeof...(CT)>(), cell);
        }

        template<std::size_t... I, class... T>
        const_reference evaluate(std::index_sequence<I...>, T... t) const
        {
            return m_f(std::get<I>(m_e).template operator()(t...)...);
        }

      private:
        std::tuple<CT...> m_e;
        functor_type m_f;
    };

    template<class F, class R, class... CT>
    template<class Func, class... CTA, class>
    inline field_function<F, R, CT...>::field_function(Func &&f,
                                                       CTA &&... e) noexcept
        : m_e(std::forward<CTA>(e)...), m_f(std::forward<Func>(f))
    {}
}

namespace xt
{
    namespace detail
    {
        template<class F, class... E>
        struct select_xfunction_expression<mure::field_expression_tag, F, E...>
        {
            using result_type = decltype(std::declval<F>()(
                std::declval<xvalue_type_t<std::decay_t<E>>>()...));
            using type = mure::field_function<F, result_type, E...>;
        };
    }
}

namespace mure
{
    using xt::operator+;
    using xt::operator-;
    using xt::operator*;
    using xt::operator/;
    using xt::operator%;
}
