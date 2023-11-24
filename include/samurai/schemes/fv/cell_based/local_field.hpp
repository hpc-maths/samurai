#pragma once
#include <xtensor/xadapt.hpp>

namespace samurai
{
    template <class field_t, class enable = void>
    class LocalField
    {
    };

    /**
     * Implementation when field size = 1
     */
    template <class field_t>
    class LocalField<field_t, std::enable_if_t<field_t::size == 1>>
    {
        using field_value_type = typename field_t::value_type;
        using cell_t           = typename field_t::cell_t;

      private:

        const cell_t& m_cell;
        field_value_type m_value; // one single value

      public:

        LocalField(const cell_t& cell, const field_value_type* data)
            : m_cell(cell)
            , m_value(*data)
        {
        }

        field_value_type& operator[]([[maybe_unused]] const cell_t& cell)
        {
            return m_value;
        }
    };

    /**
     * Implementation when field size > 1
     */
    template <class field_t>
    class LocalField<field_t, std::enable_if_t<field_t::size != 1>>
    {
        using field_value_type = typename field_t::value_type;
        using cell_t           = typename field_t::cell_t;
        // using container_t      = decltype(xt::adapt(std::declval<field_value_type*>(),
        //                                        field_t::size,
        //                                        xt::no_ownership(),
        //                                        xt::svector<size_t>{field_t::size}));
        using container_t         = decltype(xt::adapt(std::declval<const field_value_type*>(), xt::xshape<field_t::size>()));
        using container_closure_t = typename container_t::container_closure_type;

      private:

        const cell_t& m_cell;

        // xt::xarray_pointer<field_value_type> m_container;
        //   xt::xtensor_pointer<field_value_type> m_container;
        // xt::xarray_adaptor<xt::xbuffer_adaptor<double*&, xt::no_ownership, std::allocator<double>>,
        //                    xt::layout_type::row_major,
        //                    xt::svector<unsigned long, 4, std::allocator<unsigned long>, true>,
        //                    xt::xtensor_expression_tag>
        //     m_container;
        container_t m_container;
        // xt::xtensor_fixed<field_value_type, xt::xshape<field_t::size>> m_container;

      public:

        LocalField(const cell_t& cell, const field_value_type* data)
            : m_cell(cell)
            //, m_container(xt::adapt(data, field_t::size, xt::no_ownership(), xt::svector<size_t>{field_t::size}))
            //, m_container(xt::adapt(data, xt::xshape<field_t::size>()))
            , m_container(container_closure_t(data, xt::detail::fixed_compute_size<xt::xshape<field_t::size>>::value))
        {
            // auto container = xt::adapt(data, field_t::size, xt::no_ownership(), xt::svector<size_t>{1});
            // static_assert(std::is_same_v<decltype(container), void>);
            // std::copy(data, data + field_t::size, m_container.linear_begin());
            // m_container = xt::adapt(data, xt::xshape<field_t::size>());
        }

        auto operator[]([[maybe_unused]] const cell_t& cell) const
        {
            return xt::view(m_container, xt::all());
        }
    };

} // end namespace samurai
