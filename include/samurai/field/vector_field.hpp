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
#include "../field_expression.hpp"
#include "../mesh_holder.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../timers.hpp"
#include "field_base.hpp"

namespace samurai
{
    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    class VectorField;

    namespace detail
    {
        // VectorField specialization ---------------------------------------------

        template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
        struct inner_field_types<VectorField<mesh_t, value_t, n_comp, SOA>> : public crtp_field<VectorField<mesh_t, value_t, n_comp, SOA>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t                 = typename mesh_t::interval_t;
            using index_t                    = typename interval_t::index_t;
            using cell_t                     = Cell<dim, interval_t>;
            using data_type                  = field_data_storage_t<value_t, n_comp, SOA, false>;
            using local_data_type            = local_field_data_t<value_t, n_comp, SOA, false>;
            using size_type                  = typename data_type::size_type;

            static constexpr auto static_layout = data_type::static_layout;

            inline auto operator[](size_type i) const
            {
                return view(m_storage, i);
            }

            inline auto operator[](size_type i)
            {
                return view(m_storage, i);
            }

            inline auto operator[](const cell_t& cell) const
            {
                return view(m_storage, static_cast<size_type>(cell.index));
            }

            inline auto operator[](const cell_t& cell)
            {
                return view(m_storage, static_cast<size_type>(cell.index));
            }

            template <class... T>
            inline auto operator()(const std::size_t level, const interval_t& interval, const T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(const std::size_t level, const interval_t& interval, const T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                auto data = view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
#ifdef SAMURAI_CHECK_NAN

                if (xt::any(xt::isnan(data)))
                {
                    std::cerr << "READ NaN at level " << level << ", " << interval << std::endl;
                }
#endif
                return data;
            }

            template <class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage, item, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t& interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage, item, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage,
                            {item_s, item_e},
                            {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage,
                            {item_s, item_e},
                            {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class E>
            inline auto
            operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, const xt::xexpression<E>& index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);
                return view(m_storage,
                            {item_s, item_e},
                            {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class E>
            inline auto
            operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, const xt::xexpression<E>& index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);
                return view(m_storage,
                            {item_s, item_e},
                            {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            void resize()
            {
                m_storage.resize(static_cast<size_type>(this->derived_cast().mesh().nb_cells()));
                this->derived_cast().m_ghosts_updated = false;
#ifdef SAMURAI_CHECK_NAN
                m_storage.data().fill(std::nan(""));
#endif
            }

            data_type m_storage;
        };
    } // namespace detail

    // ------------------------------------------------------------------------
    // class VectorField
    // ------------------------------------------------------------------------

    template <class mesh_t_, class value_t, std::size_t n_comp_, bool SOA>
    class VectorField : public field_expression<VectorField<mesh_t_, value_t, n_comp_, SOA>>,
                        public inner_mesh_type<mesh_t_>,
                        public detail::inner_field_types<VectorField<mesh_t_, value_t, n_comp_, SOA>>,
                        public detail::FieldBase<VectorField<mesh_t_, value_t, n_comp_, SOA>>
    {
      public:

        using self_type    = VectorField<mesh_t_, value_t, n_comp_, SOA>;
        using inner_mesh_t = inner_mesh_type<mesh_t_>;
        using mesh_t       = mesh_t_;

        using value_type      = value_t;
        using inner_types     = detail::inner_field_types<self_type>;
        using data_type       = typename inner_types::data_type::container_t;
        using local_data_type = typename inner_types::local_data_type;
        using size_type       = typename inner_types::size_type;
        using inner_types::operator();
        using bc_container = std::vector<std::unique_ptr<Bc<self_type>>>;

        using inner_types::dim;
        using interval_t = typename mesh_t::interval_t;
        using cell_t     = typename inner_types::cell_t;

        using iterator               = Field_iterator<self_type, false>;
        using const_iterator         = Field_iterator<const self_type, true>;
        using reverse_iterator       = Field_reverse_iterator<iterator>;
        using const_reverse_iterator = Field_reverse_iterator<const_iterator>;

        static constexpr size_type n_comp = n_comp_;
        static constexpr bool is_soa      = SOA;
        static constexpr bool is_scalar   = false;
        using inner_types::static_layout;

        VectorField() = default;

        VectorField(std::string name, mesh_t& mesh);

        template <class E>
        VectorField(const field_expression<E>& e);

        VectorField(const VectorField&);
        VectorField& operator=(const VectorField&);

        VectorField(VectorField&&) noexcept            = default;
        VectorField& operator=(VectorField&&) noexcept = default;

        ~VectorField() = default;

        template <class E>
        VectorField& operator=(const field_expression<E>& e);

        void fill(value_type v);

      private:

        std::string m_name;

        bc_container p_bc;
        bool m_ghosts_updated = false;

        friend struct detail::inner_field_types<VectorField<mesh_t, value_t, n_comp_, SOA>>;
        friend class detail::FieldBase<VectorField<mesh_t, value_t, n_comp_, SOA>>;
    };

    // VectorField constructors -----------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline VectorField<mesh_t, value_t, n_comp_, SOA>::VectorField(std::string name, mesh_t& mesh)
        : inner_mesh_t(mesh)
        , inner_types()
        , m_name(std::move(name))
    {
        this->resize();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    template <class E>
    inline VectorField<mesh_t, value_t, n_comp_, SOA>::VectorField(const field_expression<E>& e)
        : inner_mesh_t(detail::extract_mesh(e.derived_cast()))
    {
        this->resize();
        *this = e;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline VectorField<mesh_t, value_t, n_comp_, SOA>::VectorField(const VectorField& field)
        : inner_mesh_t(field.mesh())
        , inner_types(field)
        , m_name(field.m_name)
    {
        this->copy_bc_from(field);
    }

    // VectorField operators --------------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::operator=(const VectorField& field) -> VectorField&
    {
        times::timers.start("field expressions");
        inner_mesh_t::operator=(field.mesh());
        m_name = field.m_name;
        inner_types::operator=(field);

        bc_container tmp;
        std::transform(field.p_bc.cbegin(),
                       field.p_bc.cend(),
                       std::back_inserter(tmp),
                       [](const auto& v)
                       {
                           return v->clone();
                       });
        std::swap(p_bc, tmp);
        m_ghosts_updated = field.m_ghosts_updated;
        times::timers.stop("field expressions");
        return *this;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    template <class E>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::operator=(const field_expression<E>& e) -> VectorField&
    {
        times::timers.start("field expressions");
        for_each_interval(this->mesh(),
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              noalias((*this)(level, i, index)) = e.derived_cast()(level, i, index);
                          });
        m_ghosts_updated = false;
        times::timers.stop("field expressions");
        return *this;
    }

    // VectorField methods (type-specific) ------------------------------------

    // --- fill ---------------------------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline void VectorField<mesh_t, value_t, n_comp_, SOA>::fill(value_type v)
    {
        this->m_storage.data().fill(v);
        m_ghosts_updated = false;
    }

    // VectorField helper functions -------------------------------------------

    /**
     * @brief Creates a VectorField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    [[deprecated("Use make_vector_field() instead")]] auto
    make_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        return make_vector_field<value_t, n_comp, SOA>(name, mesh, std::forward<Func>(f), gl);
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    [[deprecated("Use make_vector_field() instead")]] auto
    make_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        return make_vector_field<n_comp, SOA>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a VectorField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t,
              std::size_t n_comp,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        return make_vector_field<value_t, n_comp, SOA>(name, mesh, std::forward<Func>(f));
    }

    template <std::size_t n_comp,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        return make_vector_field<n_comp, SOA>(name, mesh, std::forward<Func>(f));
    }

    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t>
    auto make_vector_field(std::string const& name, mesh_t& mesh)
    {
        using field_t = VectorField<mesh_t, value_t, n_comp, SOA>;
        field_t f(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        if constexpr (std::is_floating_point_v<value_t>)
        {
            f.fill(static_cast<value_t>(std::nan("")));
        }
#endif
        return f;
    }

    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t>
    auto make_vector_field(std::string const& name, mesh_t& mesh, value_t init_value)
    {
        using field_t = VectorField<mesh_t, value_t, n_comp, SOA>;
        auto field    = field_t(name, mesh);
        field.fill(init_value);
        return field;
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t>
    auto make_vector_field(std::string const& name, mesh_t& mesh)
    {
        using default_value_t = double;
        return make_vector_field<default_value_t, n_comp, SOA>(name, mesh);
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t>
    auto make_vector_field(std::string const& name, mesh_t& mesh, double init_value)
    {
        using default_value_t = double;
        return make_vector_field<default_value_t, n_comp, SOA>(name, mesh, init_value);
    }

    /**
     * @brief Creates a VectorField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_vector_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        auto field = make_vector_field<value_t, n_comp, SOA, mesh_t>(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        f.fill(std::nan(""));
#else
        field.fill(0);
#endif

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const double& h = cell.length;
                          field[cell]     = gl.template quadrature<n_comp>(cell, f) / pow(h, mesh_t::dim);
                      });
        return field;
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_vector_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        using default_value_t = double;
        return make_vector_field<default_value_t, n_comp, SOA>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a VectorField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t,
              std::size_t n_comp,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    auto make_vector_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        auto field = make_vector_field<value_t, n_comp, SOA, mesh_t>(name, mesh);
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

    template <std::size_t n_comp,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    auto make_vector_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        using default_value_t = double;
        return make_vector_field<default_value_t, n_comp, SOA>(name, mesh, std::forward<Func>(f));
    }

    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh)
    {
        return make_vector_field<value_t, n_comp>(name, mesh);
    }

    template <class value_t, std::size_t n_comp, bool SOA = false, class mesh_t>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, value_t init_value)
    {
        return make_vector_field<value_t, n_comp>(name, mesh, init_value);
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh)
    {
        return make_vector_field<n_comp>(name, mesh);
    }

    template <std::size_t n_comp, bool SOA = false, class mesh_t>
    [[deprecated("Use make_vector_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, double init_value)
    {
        return make_vector_field<n_comp>(name, mesh, init_value);
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline bool operator==(const VectorField<mesh_t, value_t, n_comp_, SOA>& field1, const VectorField<mesh_t, value_t, n_comp_, SOA>& field2)
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
                              if (xt::any(xt::not_equal(field1[cell], field2[cell])))
                              {
                                  is_same = false;
                              }
                          }
                          else
                          {
                              if (xt::any(xt::abs(field1[cell] - field2[cell]) > 1e-15))
                              {
                                  is_same = false;
                              }
                          }
                      });

        return is_same;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline bool operator!=(const VectorField<mesh_t, value_t, n_comp_, SOA>& field1, const VectorField<mesh_t, value_t, n_comp_, SOA>& field2)
    {
        return !(field1 == field2);
    }

} // namespace samurai
