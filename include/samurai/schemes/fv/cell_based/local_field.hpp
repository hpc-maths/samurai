#pragma once
#include <xtensor/containers/xadapt.hpp>

namespace samurai
{
    template <class field_t, class enable = void>
    class LocalField
    {
    };

    /**
     * Implementation when number of components of field = 1
     */
    template <class field_t>
    class LocalField<field_t, std::enable_if_t<field_t::is_scalar>>
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

        const field_value_type& operator[]([[maybe_unused]] const cell_t& cell) const
        {
            assert(cell.index == m_cell.index);
            return m_value;
        }
    };

    /**
     * Implementation when number of components of field > 1
     */
    template <class field_t>
    class LocalField<field_t, std::enable_if_t<!field_t::is_scalar>>
    {
        using field_value_type    = typename field_t::value_type;
        using cell_t              = typename field_t::cell_t;
        using container_t         = decltype(xt::adapt(std::declval<const field_value_type*>(), xt::xshape<field_t::n_comp>()));
        using container_closure_t = typename container_t::container_closure_type;

      private:

        const cell_t& m_cell;
        container_t m_container;

      public:

        // cppcheck-suppress uninitMemberVar
        LocalField(const cell_t& cell, const field_value_type* data)
            : m_cell(cell)
            , m_container(container_closure_t(data, xt::detail::fixed_compute_size<xt::xshape<field_t::n_comp>>::value))
        {
        }

        auto operator[]([[maybe_unused]] const cell_t& cell) const
        {
            assert(cell.index == m_cell.index);
            return xt::view(m_container, xt::all());
        }
    };

} // end namespace samurai
