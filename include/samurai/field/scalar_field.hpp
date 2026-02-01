// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <string>
#include <type_traits>

#include "../algorithm.hpp"
#include "../concepts.hpp"
#include "../field_expression.hpp"
#include "../mesh_holder.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "access_base.hpp"
#include "field_base.hpp"

namespace samurai
{
    template <class mesh_t, class value_t>
    class ScalarField;

    namespace detail
    {
        template <class mesh_t_, class value_t>
        struct inner_field_types<ScalarField<mesh_t_, value_t>>
        {
            using mesh_t                        = mesh_t_;
            static constexpr std::size_t dim    = mesh_t::dim;
            using interval_t                    = typename mesh_t::interval_t;
            using value_type                    = value_t;
            using index_t                       = typename interval_t::index_t;
            using interval_value_t              = typename interval_t::value_t;
            using cell_t                        = typename mesh_t::cell_t;
            using data_type                     = field_data_storage_t<value_t, 1>;
            using local_data_type               = local_field_data_t<value_t, 1, false, true>;
            using size_type                     = typename data_type::size_type;
            static constexpr auto static_layout = data_type::static_layout;
        };

        // ScalarField specialization ---------------------------------------------
        template <class mesh_t, class value_t>
        struct field_data_access<ScalarField<mesh_t, value_t>> : public field_data_access_base<ScalarField<mesh_t, value_t>>
        {
            using base_type = field_data_access_base<ScalarField<mesh_t, value_t>>;
            using data_type = typename base_type::data_type;
            using size_type = typename data_type::size_type;
            using cell_t    = typename base_type::cell_t;

            using base_type::operator();

            SAMURAI_INLINE const value_t& operator[](size_type i) const
            {
                return this->storage().data()[i];
            }

            SAMURAI_INLINE value_t& operator[](size_type i)
            {
                return this->storage().data()[i];
            }

            SAMURAI_INLINE const value_t& operator[](const cell_t& cell) const
            {
                return this->storage().data()[static_cast<size_type>(cell.index)];
            }

            SAMURAI_INLINE value_t& operator[](const cell_t& cell)
            {
                return this->storage().data()[static_cast<size_type>(cell.index)];
            }
        };
    } // namespace detail

    // ------------------------------------------------------------------------
    // class ScalarField
    // ------------------------------------------------------------------------

    template <class mesh_t, class value_t = double>
    class ScalarField : public field_expression<ScalarField<mesh_t, value_t>>,
                        public inner_mesh_type<mesh_t>,
                        public detail::field_data_access<ScalarField<mesh_t, value_t>>,
                        public detail::FieldBase<ScalarField<mesh_t, value_t>>
    {
      public:

        using self_type = ScalarField<mesh_t, value_t>;

        using inner_mesh_t     = inner_mesh_type<mesh_t>;
        using data_access_type = detail::field_data_access<self_type>;
        using size_type        = typename data_access_type::size_type;

        static constexpr size_type n_comp = 1;
        static constexpr bool is_scalar   = true;

        ScalarField() = default;

        explicit ScalarField(std::string name, mesh_t& mesh);

        template <class E>
        ScalarField(const field_expression<E>& e);
        template <class E>
        ScalarField& operator=(const field_expression<E>& e);

        ScalarField(const ScalarField&);
        ScalarField& operator=(const ScalarField&);

        ScalarField(ScalarField&&) noexcept            = default;
        ScalarField& operator=(ScalarField&&) noexcept = default;

        ~ScalarField() = default;
    };

    // ScalarField constructors -----------------------------------------------

    template <class mesh_t, class value_t>
    SAMURAI_INLINE ScalarField<mesh_t, value_t>::ScalarField(std::string name, mesh_t& mesh)
        : inner_mesh_t(mesh)
    {
        this->m_name = std::move(name);
        this->resize();
    }

    template <class mesh_t, class value_t>
    template <class E>
    SAMURAI_INLINE ScalarField<mesh_t, value_t>::ScalarField(const field_expression<E>& e)
        : inner_mesh_t(detail::extract_mesh(e.derived_cast()))
    {
        this->resize();
        *this = e;
    }

    template <class mesh_t, class value_t>
    SAMURAI_INLINE ScalarField<mesh_t, value_t>::ScalarField(const ScalarField& field)
    : inner_mesh_t(field.mesh())
    {
        this->assign_from(field);
    }

    // ScalarField operators --------------------------------------------------

    template <class mesh_t, class value_t>
    SAMURAI_INLINE auto ScalarField<mesh_t, value_t>::operator=(const ScalarField& field) -> ScalarField&
    {
        this->assign_from(field);
        return *this;
    }

    template <class mesh_t, class value_t>
    template <class E>
    SAMURAI_INLINE auto ScalarField<mesh_t, value_t>::operator=(const field_expression<E>& e) -> ScalarField&
    {
        this->assign_expression(e);
        return *this;
    }

    // ScalarField helper functions -------------------------------------------

    namespace detail
    {
        template <class value_t, class mesh_t>
        auto make_field_with_nan_init(const std::string& name, mesh_t& mesh)
        {
            ScalarField<mesh_t, value_t> field(name, mesh);
#ifdef SAMURAI_CHECK_NAN
            if constexpr (std::is_floating_point_v<value_t>)
            {
                field.fill(static_cast<value_t>(std::nan("")));
            }
#endif
            return field;
        }
    }

    template <class value_t = double, class mesh_t>
        requires mesh_like<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh)
    {
        return detail::make_field_with_nan_init<value_t>(name, mesh);
    }

    template <class value_t = double, class mesh_t>
        requires mesh_like<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, value_t init_value)
    {
        auto field = detail::make_field_with_nan_init<value_t>(name, mesh);
        field.fill(init_value);
        return field;
    }

    template <class value_t = double, class mesh_t, class Func, std::size_t polynomial_degree>
        requires mesh_like<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        auto field = detail::make_field_with_nan_init<value_t>(name, mesh);
        field.fill(0);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          field[cell] = gl.template quadrature<1>(cell, f) / pow(cell.length, mesh_t::dim);
                      });
        return field;
    }

    template <class value_t = double, class mesh_t, class Func>
        requires mesh_like<mesh_t> && std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f)
    {
        auto field = detail::make_field_with_nan_init<value_t>(name, mesh);
        field.fill(0);

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          field[cell] = f(cell.center());
                      });
        return field;
    }
} // namespace samurai
