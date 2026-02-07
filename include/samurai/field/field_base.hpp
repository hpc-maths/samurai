// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "../algorithm.hpp"
#include "../bc/bc.hpp"
#include "../field_expression.hpp"
#include "../storage/containers.hpp"
#include "../timers.hpp"
#include "field_iterator.hpp"

namespace samurai
{
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

            Derived& derived_cast() & noexcept;
            const Derived& derived_cast() const& noexcept;
            Derived derived_cast() && noexcept;

          public:

            using iterator               = Field_iterator<derived_t, false>;
            using const_iterator         = Field_iterator<const derived_t, true>;
            using reverse_iterator       = Field_reverse_iterator<iterator>;
            using const_reverse_iterator = Field_reverse_iterator<const_iterator>;

            // --- element access helpers -----------------------------------------

            template <class... T>
            const auto& get_interval(std::size_t level, const auto& interval, const T... index) const;

            template <class E>
            const auto& get_interval(std::size_t level, const auto& interval, const xt::xexpression<E>& index) const;

            // --- assignment helper -------------------------------------------

            Derived& assign_from(const Derived& other);

            template <class E>
            Derived& assign_expression(const field_expression<E>& e);

            // ================================================================
            // METADATA ACCESSORS
            // ================================================================

            const std::string& name() const&;
            std::string_view name_view() const noexcept;
            std::string& name() &;

            bool& ghosts_updated();
            bool ghosts_updated() const;

            auto& array();
            const auto& array() const;

            // ================================================================
            // BOUNDARY CONDITION METHODS
            // ================================================================

            template <class Bc_derived>
            auto attach_bc(const Bc_derived& bc);

            auto& get_bc();
            const auto& get_bc() const;

            void copy_bc_from(const Derived& other);

            // ================================================================
            // ITERATOR METHODS
            // ================================================================

            auto begin();
            auto end();
            auto begin() const;
            auto end() const;
            auto cbegin() const;
            auto cend() const;
            auto rbegin();
            auto rend();
            auto rbegin() const;
            auto rend() const;
            auto rcbegin() const;
            auto rcend() const;

            // ================================================================
            // STREAM AND COMPARISON OPERATORS
            // ================================================================

            void to_stream(std::ostream& os) const;

            friend std::ostream& operator<<(std::ostream& out, const Derived& field)
            {
                field.to_stream(out);
                return out;
            }
        };

        // ====================================================================
        // FieldBase method definitions
        // ====================================================================

        // --- Protected methods ----------------------------------------------

        template <class Derived>
        SAMURAI_INLINE Derived& FieldBase<Derived>::derived_cast() & noexcept
        {
            return *static_cast<Derived*>(this);
        }

        template <class Derived>
        SAMURAI_INLINE const Derived& FieldBase<Derived>::derived_cast() const& noexcept
        {
            return *static_cast<const Derived*>(this);
        }

        template <class Derived>
        SAMURAI_INLINE Derived FieldBase<Derived>::derived_cast() && noexcept
        {
            return std::move(*static_cast<Derived*>(this));
        }

        // --- Element access helpers -----------------------------------------

        template <class Derived>
        template <class... T>
        SAMURAI_INLINE const auto& FieldBase<Derived>::get_interval(std::size_t level, const auto& interval, const T... index) const
        {
            const auto& interval_tmp = this->derived_cast().mesh().get_interval(level, interval, index...);

            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
            {
                std::ostringstream idx_ss;
                ((idx_ss << index << ' '), ...);
                auto idx_str = idx_ss.str();
                throw std::out_of_range(fmt::format("Field '{}' interval query failed on level {}: requested interval {} "
                                                    "could not be found for indices [{}]; available interval: {}",
                                                    this->derived_cast().name(),
                                                    level,
                                                    interval,
                                                    idx_str,
                                                    interval_tmp));
            }

            return interval_tmp;
        }

        template <class Derived>
        template <class E>
        SAMURAI_INLINE const auto&
        FieldBase<Derived>::get_interval(std::size_t level, const auto& interval, const xt::xexpression<E>& index) const
        {
            const auto& interval_tmp = this->derived_cast().mesh().get_interval(level, interval, index);

            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
            {
                throw std::out_of_range(fmt::format("Field '{}' interval query failed on level {}: requested interval {} "
                                                    "could not be found; available interval: {}",
                                                    this->derived_cast().name(),
                                                    level,
                                                    interval,
                                                    interval_tmp));
            }

            return interval_tmp;
        }

        // --- Assignment helpers ---------------------------------------------

        template <class Derived>
        SAMURAI_INLINE Derived& FieldBase<Derived>::assign_from(const Derived& other)
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

        template <class Derived>
        template <class E>
        SAMURAI_INLINE Derived& FieldBase<Derived>::assign_expression(const field_expression<E>& e)
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

        // --- Metadata accessors ---------------------------------------------

        template <class Derived>
        SAMURAI_INLINE const std::string& FieldBase<Derived>::name() const&
        {
            return m_name;
        }

        template <class Derived>
        SAMURAI_INLINE std::string_view FieldBase<Derived>::name_view() const noexcept
        {
            return m_name;
        }

        template <class Derived>
        SAMURAI_INLINE std::string& FieldBase<Derived>::name() &
        {
            return m_name;
        }

        template <class Derived>
        SAMURAI_INLINE bool& FieldBase<Derived>::ghosts_updated()
        {
            return m_ghosts_updated;
        }

        template <class Derived>
        SAMURAI_INLINE bool FieldBase<Derived>::ghosts_updated() const
        {
            return m_ghosts_updated;
        }

        template <class Derived>
        SAMURAI_INLINE auto& FieldBase<Derived>::array()
        {
            return this->derived_cast().storage().data();
        }

        template <class Derived>
        SAMURAI_INLINE const auto& FieldBase<Derived>::array() const
        {
            return this->derived_cast().storage().data();
        }

        // --- Boundary condition methods -------------------------------------

        template <class Derived>
        template <class Bc_derived>
        SAMURAI_INLINE auto FieldBase<Derived>::attach_bc(const Bc_derived& bc)
        {
            if (bc.stencil_size() > this->derived_cast().mesh().cfg().max_stencil_size())
            {
                std::cerr << "The stencil size required by this boundary condition (" << bc.stencil_size()
                          << ") is larger than the max_stencil_size parameter of the mesh ("
                          << this->derived_cast().mesh().cfg().max_stencil_size() << ").\nYou can set it with mesh_config.max_stencil_radius("
                          << bc.stencil_size() / 2 << ") or mesh_config.max_stencil_size(" << bc.stencil_size() << ")." << std::endl;
                exit(EXIT_FAILURE);
            }
            p_bc.push_back(bc.clone());
            return p_bc.back().get();
        }

        template <class Derived>
        SAMURAI_INLINE auto& FieldBase<Derived>::get_bc()
        {
            return p_bc;
        }

        template <class Derived>
        SAMURAI_INLINE const auto& FieldBase<Derived>::get_bc() const
        {
            return p_bc;
        }

        template <class Derived>
        SAMURAI_INLINE void FieldBase<Derived>::copy_bc_from(const Derived& other)
        {
            std::transform(other.get_bc().cbegin(),
                           other.get_bc().cend(),
                           std::back_inserter(p_bc),
                           [](const auto& v)
                           {
                               return v->clone();
                           });
        }

        // --- Iterator methods -----------------------------------------------

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::begin()
        {
            using mesh_id_t = typename derived_t::mesh_t::mesh_id_t;
            return iterator(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cbegin());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::end()
        {
            using mesh_id_t = derived_t::mesh_t::mesh_id_t;
            return iterator(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cend());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::begin() const
        {
            return cbegin();
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::end() const
        {
            return cend();
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::cbegin() const
        {
            using mesh_id_t = derived_t::mesh_t::mesh_id_t;
            return const_iterator(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cbegin());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::cend() const
        {
            using mesh_id_t = derived_t::mesh_t::mesh_id_t;
            return const_iterator(&this->derived_cast(), this->derived_cast().mesh()[mesh_id_t::cells].cend());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rbegin()
        {
            return reverse_iterator(end());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rend()
        {
            return reverse_iterator(begin());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rbegin() const
        {
            return rcbegin();
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rend() const
        {
            return rcend();
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rcbegin() const
        {
            return const_reverse_iterator(cend());
        }

        template <class Derived>
        SAMURAI_INLINE auto FieldBase<Derived>::rcend() const
        {
            return const_reverse_iterator(cbegin());
        }

        // --- Stream operators -----------------------------------------------

        template <class Derived>
        SAMURAI_INLINE void FieldBase<Derived>::to_stream(std::ostream& os) const
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

    } // namespace detail

    template <class Field>
        requires std::is_base_of_v<detail::FieldBase<Field>, Field>
    SAMURAI_INLINE bool operator==(const Field& field1, const Field& field2)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

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
                          if constexpr (std::is_integral_v<typename Field::value_type>)
                          {
                              if constexpr (Field::is_scalar)
                              {
                                  if (field1[cell] != field2[cell])
                                  {
                                      is_same = false;
                                  }
                              }
                              else
                              {
                                  // TODO: This will not work if the container is not a xtensor container
                                  static_assert(is_xtensor_container<typename Field::data_type::container_t>,
                                                "Field data_type is not an xtensor expression");
                                  if (xt::any(xt::not_equal(field1[cell], field2[cell])))
                                  {
                                      is_same = false;
                                  }
                              }
                          }
                          else
                          {
                              if constexpr (Field::is_scalar)
                              {
                                  if (std::abs(field1[cell] - field2[cell]) > 1e-15)
                                  {
                                      is_same = false;
                                  }
                              }
                              else
                              {
                                  // TODO: This will not work if the container is not a xtensor container
                                  static_assert(is_xtensor_container<typename Field::data_type::container_t>,
                                                "Field data_type is not an xtensor expression");
                                  if (xt::any(xt::abs(field1[cell] - field2[cell]) > 1e-15))
                                  {
                                      is_same = false;
                                  }
                              }
                          }
                      });

        return is_same;
    }

    template <class Field>
        requires std::is_base_of_v<detail::FieldBase<Field>, Field>
    SAMURAI_INLINE bool operator!=(const Field& field1, const Field& field2)
    {
        return !(field1 == field2);
    }

} // namespace samurai
