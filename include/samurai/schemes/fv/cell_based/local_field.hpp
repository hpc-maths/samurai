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

      private:

        cell_t& m_cell;

        xt::xarray_pointer<field_value_type> m_container;

      public:

        LocalField(const cell_t& cell, const field_value_type* data)
            : m_cell(cell)
            , m_container(data)
        {
            m_container = xt::adapt(data, field_t::size, xt::no_ownership()); //, xt::xshape<1>);
        }

        field_value_type& operator[]([[maybe_unused]] const cell_t& cell)
        {
            return m_container;
        }
    };

} // end namespace samurai
