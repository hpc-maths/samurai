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
        // ------------------------------------------------------------------------
        // struct inner_field_types
        // ------------------------------------------------------------------------

        // ScalarField specialization ---------------------------------------------
        template <class mesh_t, class value_t>
        struct inner_field_types<ScalarField<mesh_t, value_t>> : public crtp_base<ScalarField<mesh_t, value_t>>
        {
            static constexpr std::size_t dim    = mesh_t::dim;
            using interval_t                    = typename mesh_t::interval_t;
            using index_t                       = typename interval_t::index_t;
            using interval_value_t              = typename interval_t::value_t;
            using cell_t                        = Cell<dim, interval_t>;
            using data_type                     = field_data_storage_t<value_t, 1>;
            using local_data_type               = local_field_data_t<value_t, 1, false, true>;
            using size_type                     = typename data_type::size_type;
            static constexpr auto static_layout = data_type::static_layout;

            inline const value_t& operator[](size_type i) const
            {
                return m_storage.data()[i];
            }

            inline value_t& operator[](size_type i)
            {
                return m_storage.data()[i];
            }

            inline const value_t& operator[](const cell_t& cell) const
            {
                return m_storage.data()[static_cast<size_type>(cell.index)];
            }

            inline value_t& operator[](const cell_t& cell)
            {
                return m_storage.data()[static_cast<size_type>(cell.index)];
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
                    for (decltype(interval_tmp.index) i = interval_tmp.index + interval.start; i < interval_tmp.index + interval.end;
                         i += interval.step)
                    {
                        if (std::isnan(this->derived_cast().m_storage.data()[static_cast<std::size_t>(i)]))
                        {
                            // std::cerr << "READ NaN at level " << level << ", in interval " << interval << std::endl;
                            auto ii   = i - interval_tmp.index;
                            auto cell = this->derived_cast().mesh().get_cell(level, static_cast<int>(ii), index...);
                            std::cerr << "READ NaN in " << cell << std::endl;
                            break;
                        }
                    }
                }
#endif
                return data;
            }

            inline auto operator()(const std::size_t level,
                                   const interval_t& interval,
                                   const xt::xtensor_fixed<interval_value_t, xt::xshape<dim - 1>>& index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);

                // std::cout << "READ OR WRITE: " << level << " " << interval << " " << (... << index) << std::endl;
                return view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(const std::size_t level,
                                   const interval_t& interval,
                                   const xt::xtensor_fixed<interval_value_t, xt::xshape<dim - 1>>& index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);
                // std::cout << "READ: " << level << " " << interval << " " << (... << index) << std::endl;
                auto data = view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});

#ifdef SAMURAI_CHECK_NAN
                if (xt::any(xt::isnan(data)))
                {
                    for (decltype(interval_tmp.index) i = interval_tmp.index + interval.start; i < interval_tmp.index + interval.end;
                         i += interval.step)
                    {
                        if (std::isnan(this->derived_cast().m_storage.data()[static_cast<std::size_t>(i)]))
                        {
                            // std::cerr << "READ NaN at level " << level << ", in interval " << interval << std::endl;
                            auto ii   = i - interval_tmp.index;
                            auto cell = this->derived_cast().mesh().get_cell(level, static_cast<int>(ii), index);
                            std::cerr << "READ NaN in " << cell << std::endl;
                            break;
                        }
                    }
                }
#endif
                return data;
            }

            void resize()
            {
                m_storage.resize(static_cast<size_type>(this->derived_cast().mesh().nb_cells()));
#ifdef SAMURAI_CHECK_NAN
                if constexpr (std::is_floating_point_v<value_t>)
                {
                    this->derived_cast().m_storage.data().fill(std::nan(""));
                }
#endif
            }

            data_type m_storage;
        };
    } // namespace detail

    /**
     * @brief Scalar field class
     *
     * A scalar field represents a single value per mesh cell.
     * This class inherits from FieldBase to reduce code duplication.
     *
     * @tparam mesh_t_ The mesh type
     * @tparam value_t The value type (default: double)
     */
    template <class mesh_t_, class value_t = double>
    class ScalarField : public FieldBase<ScalarField<mesh_t_, value_t>, mesh_t_, value_t>
    {
      public:

        using base_type = FieldBase<ScalarField<mesh_t_, value_t>, mesh_t_, value_t>;
        using self_type = ScalarField<mesh_t_, value_t>;
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

        // Scalar-specific constants
        static constexpr size_type n_comp = 1;
        static constexpr bool is_scalar   = true;

        // Constructors
        ScalarField() = default;

        ScalarField(std::string name, mesh_t& mesh)
            : base_type(std::move(name), mesh)
        {
        }

        template <class E>
        ScalarField(const field_expression<E>& e)
            : base_type(e)
        {
        }

        // Assignment from field expression
        template <class E>
        ScalarField& operator=(const field_expression<E>& e)
        {
            return base_type::operator=(e);
        }

        // Copy boundary conditions from another ScalarField
        void copy_bc_from(const ScalarField& other)
        {
            base_type::copy_bc_from(other);
        }
    };

    // ScalarField helper functions -------------------------------------------

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t, class mesh_t, class Func, std::size_t polynomial_degree>
    [[deprecated("Use make_scalar_field() instead")]] auto
    make_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        return make_scalar_field<value_t>(name, mesh, std::forward<Func>(f), gl);
    }

    template <class mesh_t, class Func, std::size_t polynomial_degree>
    [[deprecated("Use make_scalar_field() instead")]] auto
    make_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        return make_scalar_field<double>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        return make_scalar_field<value_t>(name, mesh, std::forward<Func>(f));
    }

    template <class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        return make_scalar_field<double>(name, mesh, std::forward<Func>(f));
    }

    template <class value_t, class mesh_t>
    auto make_scalar_field(std::string const& name, mesh_t& mesh)
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

    template <class value_t, class mesh_t>
    auto make_scalar_field(std::string const& name, mesh_t& mesh, value_t init_value)
    {
        using field_t = ScalarField<mesh_t, value_t>;
        auto field    = field_t(name, mesh);
        field.fill(init_value);
        return field;
    }

    template <class mesh_t>
    auto make_scalar_field(std::string const& name, mesh_t& mesh)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t>(name, mesh);
    }

    template <class mesh_t>
    auto make_scalar_field(std::string const& name, mesh_t& mesh, double init_value)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t>(name, mesh, init_value);
    }

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t, class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_scalar_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
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
                          const double& h = cell.length;
                          field[cell]     = gl.template quadrature<1>(cell, f) / pow(h, mesh_t::dim);
                      });
        return field;
    }

    template <class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_scalar_field(std::string const& name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a ScalarField.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    auto make_scalar_field(std::string const& name, mesh_t& mesh, Func&& f)
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
    auto make_scalar_field(std::string const& name, mesh_t& mesh, Func&& f)
    {
        using default_value_t = double;
        return make_scalar_field<default_value_t, mesh_t>(name, mesh, std::forward<Func>(f));
    }

    template <class value_t, class mesh_t>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh)
    {
        return make_scalar_field<value_t>(name, mesh);
    }

    template <class value_t, class mesh_t>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, value_t init_value)
    {
        return make_scalar_field<value_t>(name, mesh, init_value);
    }

    template <class mesh_t>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh)
    {
        return make_scalar_field<double>(name, mesh);
    }

    template <class mesh_t>
    [[deprecated("Use make_scalar_field() instead")]] auto make_field(std::string const& name, mesh_t& mesh, double init_value)
    {
        return make_scalar_field<double>(name, mesh, init_value);
    }

} // namespace samurai
