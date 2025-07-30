// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <filesystem>
namespace fs = std::filesystem;

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "../algorithm.hpp"
#include "../bc.hpp"
#include "../cell.hpp"
#include "../cell_array.hpp"
#include "../field_expression.hpp"
#include "../mesh_holder.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../storage/containers.hpp"
#include "../timers.hpp"
#include "field_iterator.hpp"

namespace samurai
{

    namespace detail
    {
        template <class Field, class = void>
        struct inner_field_types;
    }

    /**
     * @brief Base class for all field types (ScalarField and VectorField)
     *
     * This class contains the common functionality shared between ScalarField and VectorField,
     * including iterators, boundary conditions, name management, and basic operations.
     *
     * @tparam Derived The derived field type (CRTP pattern)
     * @tparam mesh_t The mesh type
     * @tparam value_t The value type
     */
    template <class Derived, class mesh_t_, class value_t>
    class FieldBase : public field_expression<Derived>,
                      public inner_mesh_type<mesh_t_>,
                      public detail::inner_field_types<Derived>
    {
      public:

        using self_type    = Derived;
        using inner_mesh_t = inner_mesh_type<mesh_t_>;
        using mesh_t       = mesh_t_;
        using value_type   = value_t;
        using inner_types  = detail::inner_field_types<Derived>;
        using data_type    = typename inner_types::data_type::container_t;
        using size_type    = typename inner_types::size_type;
        using bc_container = std::vector<std::unique_ptr<Bc<Derived>>>;

        using inner_types::dim;
        using interval_t = typename mesh_t::interval_t;
        using cell_t     = typename inner_types::cell_t;

        using iterator               = Field_iterator<Derived, false>;
        using const_iterator         = Field_iterator<const Derived, true>;
        using reverse_iterator       = Field_reverse_iterator<iterator>;
        using const_reverse_iterator = Field_reverse_iterator<const_iterator>;

        using inner_types::operator();
        using inner_types::static_layout;

        // Constructors
        FieldBase() = default;

        explicit FieldBase(std::string name, mesh_t& mesh)
            : inner_mesh_t(mesh)
            , inner_types()
            , m_name(std::move(name))
        {
            this->resize();
        }

        template <class E>
        explicit FieldBase(const field_expression<E>& e)
            : inner_mesh_t(detail::extract_mesh(e.derived_cast()))
        {
            this->resize();
            derived_cast() = e;
        }

        FieldBase(const FieldBase& other)
            : inner_mesh_t(other.mesh())
            , inner_types(other)
            , m_name(other.m_name)
        {
            copy_bc_from(other.derived_cast());
        }

        // Assignment operators
        FieldBase& operator=(const FieldBase& other)
        {
            if (this != &other)
            {
                times::timers.start("field expressions");
                inner_mesh_t::operator=(other.mesh());
                m_name = other.m_name;
                inner_types::operator=(other);

                bc_container tmp;
                std::transform(other.p_bc.cbegin(),
                               other.p_bc.cend(),
                               std::back_inserter(tmp),
                               [](const auto& v)
                               {
                                   return v->clone();
                               });
                std::swap(p_bc, tmp);
                times::timers.stop("field expressions");
            }
            return *this;
        }

        template <class E>
        Derived& operator=(const field_expression<E>& e)
        {
            times::timers.start("field expressions");
            for_each_interval(this->mesh(),
                              [&](std::size_t level, const auto& i, const auto& index)
                              {
                                  noalias(derived_cast()(level, i, index)) = e.derived_cast()(level, i, index);
                              });
            times::timers.stop("field expressions");
            return derived_cast();
        }

        // Move semantics
        FieldBase(FieldBase&&) noexcept            = default;
        FieldBase& operator=(FieldBase&&) noexcept = default;

        // Destructor
        virtual ~FieldBase() = default;

        // Common methods

        void fill(value_type v)
        {
            this->m_storage.data().fill(v);
        }

        const data_type& array() const
        {
            return this->m_storage.data();
        }

        data_type& array()
        {
            return this->m_storage.data();
        }

        const std::string& name() const
        {
            return m_name;
        }

        std::string& name()
        {
            return m_name;
        }

        void to_stream(std::ostream& os) const
        {
            os << "Field " << m_name << "\n";

#ifdef SAMURAI_CHECK_NAN
            using mesh_id_t = typename mesh_t::mesh_id_t;
            for_each_cell(this->mesh()[mesh_id_t::reference],
#else
            for_each_cell(this->mesh(),
#endif
                          [&](auto& cell)
                          {
                              os << "\tlevel: " << cell.level << " coords: " << cell.center() << " index: " << cell.index
                                 << ", value: " << derived_cast().operator[](cell) << "\n";
                          });
        }

        // Boundary conditions
        template <class Bc_derived>
        auto attach_bc(const Bc_derived& bc)
        {
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

        // Iterator methods
        iterator begin()
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;
            return iterator(&derived_cast(), this->mesh()[mesh_id_t::cells].cbegin());
        }

        const_iterator begin() const
        {
            return cbegin();
        }

        const_iterator cbegin() const
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;
            return const_iterator(&derived_cast(), this->mesh()[mesh_id_t::cells].cbegin());
        }

        iterator end()
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;
            return iterator(&derived_cast(), this->mesh()[mesh_id_t::cells].cend());
        }

        const_iterator end() const
        {
            return cend();
        }

        const_iterator cend() const
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;
            return const_iterator(&derived_cast(), this->mesh()[mesh_id_t::cells].cend());
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator(end());
        }

        const_reverse_iterator rbegin() const
        {
            return rcbegin();
        }

        const_reverse_iterator rcbegin() const
        {
            return const_reverse_iterator(cend());
        }

        reverse_iterator rend()
        {
            return reverse_iterator(begin());
        }

        const_reverse_iterator rend() const
        {
            return rcend();
        }

        const_reverse_iterator rcend() const
        {
            return const_reverse_iterator(cbegin());
        }

      protected:

        // Common interval access methods
        template <class... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const T... index) const
        {
            const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index...);

            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
            {
                (std::cout << ... << index) << std::endl;
                throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
            }

            return interval_tmp;
        }

        const interval_t&
        get_interval(std::size_t level, const interval_t& interval, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const
        {
            const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index);

            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
            {
                throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
            }

            return interval_tmp;
        }

        // CRTP helper methods
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
            return *static_cast<Derived*>(this);
        }

      private:

        std::string m_name;
        bc_container p_bc;

        friend struct detail::inner_field_types<Derived>;
    };

    // Common operators for all field types
    template <class Derived, class mesh_t, class value_t>
    inline std::ostream& operator<<(std::ostream& out, const FieldBase<Derived, mesh_t, value_t>& field)
    {
        field.to_stream(out);
        return out;
    }

    template <class Derived, class mesh_t, class value_t>
    inline bool operator==(const FieldBase<Derived, mesh_t, value_t>& field1, const FieldBase<Derived, mesh_t, value_t>& field2)
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;

        if (field1.mesh() != field2.mesh())
        {
            std::cout << "mesh different" << std::endl;
            return false;
        }

        auto& mesh   = field1.mesh();
        bool is_same = true;
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          if constexpr (std::is_integral_v<value_t>)
                          {
                              if (field1[cell] != field2[cell])
                              {
                                  is_same = false;
                              }
                          }
                          else
                          {
                              if (std::abs(field1[cell] - field2[cell]) > 1e-15)
                              {
                                  is_same = false;
                              }
                          }
                      });

        return is_same;
    }

    template <class Derived, class mesh_t, class value_t>
    inline bool operator!=(const FieldBase<Derived, mesh_t, value_t>& field1, const FieldBase<Derived, mesh_t, value_t>& field2)
    {
        return !(field1 == field2);
    }

} // namespace samurai
