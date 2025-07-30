// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm.hpp"
#include "../cell.hpp"
#include "../crtp.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "field_base.hpp"

namespace samurai
{
    namespace detail
    {
        // VectorField specialization ---------------------------------------------

        template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
        struct inner_field_types<VectorField<mesh_t, value_t, n_comp, SOA>> : public crtp_base<VectorField<mesh_t, value_t, n_comp, SOA>>
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
                    // std::cout << data << std::endl;
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
#ifdef SAMURAI_CHECK_NAN
                m_storage.data().fill(std::nan(""));
#endif
            }

            data_type m_storage;
        };
    } // namespace detail

    /**
     * @brief Vector field class
     *
     * A vector field represents multiple values per mesh cell.
     * This class inherits from FieldBase to reduce code duplication.
     *
     * @tparam mesh_t_ The mesh type
     * @tparam value_t The value type
     * @tparam n_comp_ Number of components
     * @tparam SOA Structure of Arrays layout (default: false for AOS)
     */
    template <class mesh_t_, class value_t, std::size_t n_comp_, bool SOA>
    class VectorField : public FieldBase<VectorField<mesh_t_, value_t, n_comp_, SOA>, mesh_t_, value_t>
    {
      public:

        using base_type = FieldBase<VectorField<mesh_t_, value_t, n_comp_, SOA>, mesh_t_, value_t>;
        using self_type = VectorField<mesh_t_, value_t, n_comp_, SOA>;
        using mesh_t    = mesh_t_;

        // Import types from base class
        using typename base_type::bc_container;
        using typename base_type::cell_t;
        using typename base_type::const_iterator;
        using typename base_type::const_reverse_iterator;
        using typename base_type::data_type;
        using typename base_type::interval_t;
        using typename base_type::iterator;
        using typename base_type::local_data_type;
        using typename base_type::reverse_iterator;
        using typename base_type::size_type;
        using typename base_type::value_type;

        // Vector-specific constants
        static constexpr size_type n_comp = n_comp_;
        static constexpr bool is_soa      = SOA;
        static constexpr bool is_scalar   = false;

        // Constructors
        VectorField() = default;

        VectorField(std::string name, mesh_t& mesh)
            : base_type(std::move(name), mesh)
        {
        }

        template <class E>
        VectorField(const field_expression<E>& e)
            : base_type(e)
        {
        }

        // Assignment from field expression
        template <class E>
        VectorField& operator=(const field_expression<E>& e)
        {
            base_type::operator=(e);
            return *this;
        }

        // Copy boundary conditions from another VectorField
        void copy_bc_from(const VectorField& other)
        {
            base_type::copy_bc_from(other);
        }
    };

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
        field.fill(std::nan(""));
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

} // namespace samurai
