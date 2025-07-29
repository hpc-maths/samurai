// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtl/xiterator_base.hpp>

#include "../cell_array.hpp"

namespace samurai
{
    // ------------------------------------------------------------------------
    // class Field_iterator
    // ------------------------------------------------------------------------

    template <class Field, bool is_const>
    class Field_iterator : public xtl::xrandom_access_iterator_base3<Field_iterator<Field, is_const>,
                                                                     CellArray_iterator<const typename Field::mesh_t::ca_type, true>>
    {
      public:

        using self_type   = Field_iterator<Field, is_const>;
        using ca_iterator = CellArray_iterator<const typename Field::mesh_t::ca_type, true>;

        using reference       = default_view_t<typename Field::data_type>;
        using difference_type = typename ca_iterator::difference_type;

        Field_iterator(Field* field, const ca_iterator& ca_it);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        auto operator*() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

      private:

        Field* p_field;
        ca_iterator m_ca_it;
    };

    // Field_iterator constructors --------------------------------------------

    template <class Field, bool is_const>
    Field_iterator<Field, is_const>::Field_iterator(Field* field, const ca_iterator& ca_it)
        : p_field(field)
        , m_ca_it(ca_it)
    {
    }

    // Field_iterator operators -----------------------------------------------

    template <class Field, bool is_const>
    inline auto Field_iterator<Field, is_const>::operator++() -> self_type&
    {
        ++m_ca_it;
        return *this;
    }

    template <class Field, bool is_const>
    inline auto Field_iterator<Field, is_const>::operator--() -> self_type&
    {
        --m_ca_it;
        return *this;
    }

    template <class Field, bool is_const>
    inline auto Field_iterator<Field, is_const>::operator+=(difference_type n) -> self_type&
    {
        m_ca_it += n;
        return *this;
    }

    template <class Field, bool is_const>
    inline auto Field_iterator<Field, is_const>::operator-=(difference_type n) -> self_type&
    {
        m_ca_it -= n;
        return *this;
    }

    template <class Field, bool is_const>
    inline auto Field_iterator<Field, is_const>::operator*() const
    {
        std::size_t level = m_ca_it.level();
        auto& index       = m_ca_it.index();
        return (*p_field)(level, *m_ca_it, index);
    }

    // Field_iterator methods -------------------------------------------------

    template <class Field, bool is_const>
    inline bool Field_iterator<Field, is_const>::equal(const self_type& rhs) const
    {
        return m_ca_it.equal(rhs.m_ca_it);
    }

    template <class Field, bool is_const>
    inline bool Field_iterator<Field, is_const>::less_than(const self_type& rhs) const
    {
        return m_ca_it.less_than(rhs.m_ca_it);
    }

    // Field_iterator extern operators ----------------------------------------

    template <class Field, bool is_const>
    inline bool operator==(const Field_iterator<Field, is_const>& it1, const Field_iterator<Field, is_const>& it2)
    {
        return it1.equal(it2);
    }

    template <class Field, bool is_const>
    inline bool operator<(const Field_iterator<Field, is_const>& it1, const Field_iterator<Field, is_const>& it2)
    {
        return it1.less_than(it2);
    }

    template <class iterator>
    class Field_reverse_iterator : public std::reverse_iterator<iterator>
    {
      public:

        using base_type = std::reverse_iterator<iterator>;

        explicit Field_reverse_iterator(iterator&& it)
            : base_type(std::move(it))
        {
        }
    };

} // namespace samurai