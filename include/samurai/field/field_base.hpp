// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "../algorithm.hpp"
#include "../bc/bc.hpp"
#include "../cell_array.hpp"
#include "../field_expression.hpp"
#include "../storage/containers.hpp"
#include "../timers.hpp"

namespace samurai
{
    template <class iterator>
    class Field_reverse_iterator : public std::reverse_iterator<iterator>
    {
      public:

        using base_type = std::reverse_iterator<iterator>;

        explicit Field_reverse_iterator(const iterator& it)
            : base_type(it)
        {
        }
    };

    namespace detail
    {
        template <class D>
        struct crtp_field
        {
            using derived_type = D;

            derived_type& derived_cast() & noexcept
            {
                return *static_cast<derived_type*>(this);
            }

            const derived_type& derived_cast() const& noexcept
            {
                return *static_cast<const derived_type*>(this);
            }

            derived_type derived_cast() && noexcept
            {
                return *static_cast<derived_type*>(this);
            }
        };
    } // namespace detail

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
        return view(p_field->storage(), {m_ca_it->index + m_ca_it->start, m_ca_it->index + m_ca_it->end});
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

    // ------------------------------------------------------------------------
    // class FieldBase - CRTP base for common field functionality
    // ------------------------------------------------------------------------

    namespace detail
    {
        template <class Derived>
        class FieldBase
        {
          protected:

            using derived_t    = Derived;
            using bc_container = std::vector<std::unique_ptr<Bc<Derived>>>;

            std::string m_name;
            bc_container p_bc;
            bool m_ghosts_updated = false;

            Derived& derived_cast() & noexcept
            {
                return *static_cast<Derived*>(this);
            }

            const Derived& derived_cast() const& noexcept
            {
                return *static_cast<const Derived*>(this);
            }

            Derived derived_cast() && noexcept
            {
                return std::move(*static_cast<Derived*>(this));
            }

          public:

            // --- element access helpers -----------------------------------------

            template <class... T>
            const auto& get_interval(std::size_t level, const auto& interval, const T... index) const
            {
                const auto& interval_tmp = this->derived_cast().mesh().get_interval(level, interval, index...);

                if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
                {
                    std::ostringstream idx_ss;
                    ((idx_ss << index << ' '), ...);
                    auto idx_str = idx_ss.str();
                    throw std::out_of_range(
                        fmt::format("FIELD ERROR on level {}: try to find interval {} (indices: {})", level, interval, idx_str));
                }

                return interval_tmp;
            }

            template <class E>
            const auto& get_interval(std::size_t level, const auto& interval, const xt::xexpression<E>& index) const
            {
                const auto& interval_tmp = this->derived_cast().mesh().get_interval(level, interval, index);

                if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
                {
                    throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
                }

                return interval_tmp;
            }

            // --- assignment helper -------------------------------------------

            Derived& assign_from(const Derived& other)
            {
                if (this == &other)
                {
                    return this->derived_cast();
                }

                times::timers.start("field expressions");

                using inner_mesh_t  = typename Derived::inner_mesh_t;
                using data_access_t = typename Derived::data_access_type;

                static_cast<inner_mesh_t&>(this->derived_cast())  = other.mesh();
                m_name                                            = other.m_name;
                static_cast<data_access_t&>(this->derived_cast()) = other;
                bc_container tmp;
                std::transform(other.p_bc.cbegin(),
                               other.p_bc.cend(),
                               std::back_inserter(tmp),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });
                std::swap(p_bc, tmp);
                m_ghosts_updated = other.m_ghosts_updated;

                times::timers.stop("field expressions");
                return this->derived_cast();
            }

            template <class E>
            Derived& assign_expression(const field_expression<E>& e)
            {
                times::timers.start("field expressions");
                for_each_interval(this->derived_cast().mesh(),
                                  [&](std::size_t level, const auto& i, const auto& index)
                                  {
                                      noalias(this->derived_cast()(level, i, index)) = e.derived_cast()(level, i, index);
                                  });
                m_ghosts_updated = false;
                times::timers.stop("field expressions");
                return this->derived_cast();
            }

          public:

            // ================================================================
            // METADATA ACCESSORS
            // ================================================================

            const std::string& name() const
            {
                return m_name;
            }

            std::string& name()
            {
                return m_name;
            }

            bool& ghosts_updated()
            {
                return m_ghosts_updated;
            }

            bool ghosts_updated() const
            {
                return m_ghosts_updated;
            }

            auto& array()
            {
                return this->derived_cast().storage().data();
            }

            const auto& array() const
            {
                return this->derived_cast().storage().data();
            }

            // ================================================================
            // BOUNDARY CONDITION METHODS
            // ================================================================

            template <class Bc_derived>
            auto attach_bc(const Bc_derived& bc)
            {
                if (bc.stencil_size() > this->derived_cast().mesh().cfg().max_stencil_size())
                {
                    std::cerr << "The stencil size required by this boundary condition (" << bc.stencil_size()
                              << ") is larger than the max_stencil_size parameter of the mesh ("
                              << this->derived_cast().mesh().cfg().max_stencil_size()
                              << ").\nYou can set it with mesh_config.max_stencil_radius(" << bc.stencil_size() / 2
                              << ") or mesh_config.max_stencil_size(" << bc.stencil_size() << ")." << std::endl;
                    exit(EXIT_FAILURE);
                }
                p_bc.push_back(bc.clone());
                return p_bc.back().get();
            }

            auto& get_bc()
            {
                return p_bc;
            }

            const auto& get_bc() const
            {
                return p_bc;
            }

            void copy_bc_from(const Derived& other)
            {
                std::transform(other.get_bc().cbegin(),
                               other.get_bc().cend(),
                               std::back_inserter(p_bc),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });
            }

            // ================================================================
            // ITERATOR METHODS
            // ================================================================

            auto begin()
            {
                using mesh_id_t = typename std::remove_reference_t<decltype(this->derived_cast().mesh())>::mesh_id_t;
                return Field_iterator<Derived, false>(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cbegin());
            }

            auto end()
            {
                using mesh_id_t = typename std::remove_reference_t<decltype(this->derived_cast().mesh())>::mesh_id_t;
                return Field_iterator<Derived, false>(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cend());
            }

            auto begin() const
            {
                return cbegin();
            }

            auto end() const
            {
                return cend();
            }

            auto cbegin() const
            {
                using mesh_id_t = typename std::remove_reference_t<decltype(this->derived_cast().mesh())>::mesh_id_t;
                return Field_iterator<const Derived, true>(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cbegin());
            }

            auto cend() const
            {
                using mesh_id_t = typename std::remove_reference_t<decltype(this->derived_cast().mesh())>::mesh_id_t;
                return Field_iterator<const Derived, true>(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cend());
            }

            auto rbegin()
            {
                return typename Derived::reverse_iterator(end());
            }

            auto rend()
            {
                return typename Derived::reverse_iterator(begin());
            }

            auto rbegin() const
            {
                return rcbegin();
            }

            auto rend() const
            {
                return rcend();
            }

            auto rcbegin() const
            {
                return typename Derived::const_reverse_iterator(cend());
            }

            auto rcend() const
            {
                return typename Derived::const_reverse_iterator(cbegin());
            }

            // ================================================================
            // STREAM AND COMPARISON OPERATORS (moved to derived)
            // ================================================================

            void to_stream(std::ostream& os) const
            {
                os << "Field " << m_name << "\n";

#ifdef SAMURAI_CHECK_NAN
                using mesh_id_t = typename std::remove_reference_t<decltype(this->derived_cast().mesh())>::mesh_id_t;
                for_each_cell(this->derived_cast().mesh()[mesh_id_t::reference],
#else
                for_each_cell(this->derived_cast().mesh(),
#endif
                              [&](auto& cell)
                              {
                                  os << "\tlevel: " << cell.level << " coords: " << cell.center() << " index: " << cell.index
                                     << ", value: " << this->derived_cast().operator[](cell) << "\n";
                              });
            }

            friend std::ostream& operator<<(std::ostream& out, const Derived& field)
            {
                field.to_stream(out);
                return out;
            }
        };
    } // namespace detail
} // namespace samurai
