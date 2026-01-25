// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "../algorithm.hpp"
#include "../bc/bc.hpp"
#include "../cell.hpp"
#include "../concepts.hpp"
#include "../field_expression.hpp"
#include "../mesh_holder.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../timers.hpp"
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
            using cell_t                        = Cell<dim, interval_t>;
            using data_type                     = field_data_storage_t<value_t, 1>;
            using local_data_type               = local_field_data_t<value_t, 1, false, true>;
            using size_type                     = typename data_type::size_type;
            static constexpr auto static_layout = data_type::static_layout;
        };

        // ScalarField specialization ---------------------------------------------
        template <class mesh_t, class value_t>
        struct field_data_access<ScalarField<mesh_t, value_t>> : public field_data_access_base<ScalarField<mesh_t, value_t>>
        {
            using base_type       = field_data_access_base<ScalarField<mesh_t, value_t>>;
            using data_type       = typename base_type::data_type;
            using value_type      = value_t;
            using local_data_type = typename base_type::local_data_type;
            using size_type       = typename data_type::size_type;
            using cell_t          = typename base_type::cell_t;
            using interval_t      = typename base_type::interval_t;
            using index_t         = typename base_type::index_t;

            using base_type::static_layout;

            using base_type::operator();

            inline const value_t& operator[](size_type i) const
            {
                return this->storage().data()[i];
            }

            inline value_t& operator[](size_type i)
            {
                return this->storage().data()[i];
            }

            inline const value_t& operator[](const cell_t& cell) const
            {
                return this->storage().data()[static_cast<size_type>(cell.index)];
            }

            inline value_t& operator[](const cell_t& cell)
            {
                return this->storage().data()[static_cast<size_type>(cell.index)];
            }
        };
    } // namespace detail

    // ------------------------------------------------------------------------
    // class ScalarField
    // ------------------------------------------------------------------------

    template <class mesh_t_, class value_t = double>
    class ScalarField : public field_expression<ScalarField<mesh_t_, value_t>>,
                        public inner_mesh_type<mesh_t_>,
                        public detail::field_data_access<ScalarField<mesh_t_, value_t>>,
                        public detail::FieldBase<ScalarField<mesh_t_, value_t>>
    {
      public:

        using self_type  = ScalarField<mesh_t_, value_t>;
        using mesh_t     = mesh_t_;
        using value_type = value_t;

        using inner_mesh_t     = inner_mesh_type<mesh_t_>;
        using data_access_type = detail::field_data_access<self_type>;
        using index_t          = typename data_access_type::index_t;
        using size_type        = typename data_access_type::size_type;
        using local_data_type  = typename data_access_type::local_data_type;
        using cell_t           = typename data_access_type::cell_t;
        using interval_t       = typename data_access_type::interval_t;
        using data_access_type::dim;

        // using bc_container = std::vector<std::unique_ptr<Bc<self_type>>>;

        // using data_access_type::dim;
        // using interval_t = typename mesh_t::interval_t;
        // using cell_t     = typename data_access_type::cell_t;

        using iterator               = Field_iterator<self_type, false>;
        using const_iterator         = Field_iterator<const self_type, true>;
        using reverse_iterator       = Field_reverse_iterator<iterator>;
        using const_reverse_iterator = Field_reverse_iterator<const_iterator>;

        static constexpr size_type n_comp = 1;
        static constexpr bool is_scalar   = true;
        using data_access_type::static_layout;

        ScalarField() = default;

        explicit ScalarField(std::string name, mesh_t& mesh);

        ScalarField(const ScalarField&);
        ScalarField& operator=(const ScalarField&);

        ScalarField(ScalarField&&) noexcept            = default;
        ScalarField& operator=(ScalarField&&) noexcept = default;

        ~ScalarField() = default;

        template <class E>
        ScalarField(const field_expression<E>& e);
        template <class E>
        ScalarField& operator=(const field_expression<E>& e);
    };

    // ScalarField constructors -----------------------------------------------

    template <class mesh_t, class value_t>
    inline ScalarField<mesh_t, value_t>::ScalarField(std::string name, mesh_t& mesh)
        : inner_mesh_t(mesh)
    {
        this->m_name = std::move(name);
        this->resize();
    }

    template <class mesh_t, class value_t>
    template <class E>
    inline ScalarField<mesh_t, value_t>::ScalarField(const field_expression<E>& e)
        : inner_mesh_t(detail::extract_mesh(e.derived_cast()))
    {
        this->resize();
        *this = e;
    }

    template <class mesh_t, class value_t>
    inline ScalarField<mesh_t, value_t>::ScalarField(const ScalarField& field)
    {
        this->assign_from(field);
    }

    // ScalarField operators --------------------------------------------------

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::operator=(const ScalarField& field) -> ScalarField&
    {
        this->assign_from(field);
        return *this;
    }

    template <class mesh_t, class value_t>
    template <class E>
    inline auto ScalarField<mesh_t, value_t>::operator=(const field_expression<E>& e) -> ScalarField&
    {
        this->assign_expression(e);
        return *this;
    }

    // ScalarField helper functions -------------------------------------------

    template <class value_t = double, class mesh_t>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh)
    {
        using field_t = ScalarField<mesh_t, value_t>;
        field_t f(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        if constexpr (std::is_floating_point_v<value_t>)
        {
            f.fill(static_cast<value_t>(std::nan("")));
        }
#endif
        return f;
    }

    template <class value_t = double, class mesh_t>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, value_t init_value)
    {
        auto field = make_scalar_field<value_t, mesh_t>(name, mesh);
        field.fill(init_value);
        return field;
    }

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t = double, class mesh_t, class Func, std::size_t polynomial_degree>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        auto field = make_scalar_field<value_t, mesh_t>(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        f.fill(std::nan(""));
#else
        field.fill(0);
#endif

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const double& h = cell.length;
                          field[cell]     = gl.template quadrature<1>(cell, f) / pow(h, mesh_t::dim);
                      });
        return field;
    }

    template <class mesh_t, class Func, std::size_t polynomial_degree>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t = double,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f)
    {
        auto field = make_scalar_field<value_t, mesh_t>(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        field.fill(std::nan(""));
#else
        field.fill(0);
#endif

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          field[cell] = f(cell.center());
                      });
        return field;
    }

    template <class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
        requires IsMesh<mesh_t>
    auto make_scalar_field(const std::string& name, mesh_t& mesh, Func&& f)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t>(name, mesh, std::forward<Func>(f));
    }

    template <class mesh_t, class value_t>
    inline bool operator==(const ScalarField<mesh_t, value_t>& field1, const ScalarField<mesh_t, value_t>& field2)
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

    template <class mesh_t, class value_t>
    inline bool operator!=(const ScalarField<mesh_t, value_t>& field1, const ScalarField<mesh_t, value_t>& field2)
    {
        return !(field1 == field2);
    }
} // namespace samurai
