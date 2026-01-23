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
    template <class mesh_t, class value_t>
    class ScalarField;

    namespace detail
    {
        template <class Field, class = void>
        struct inner_field_types;

        // ScalarField specialization ---------------------------------------------
        template <class mesh_t, class value_t>
        struct inner_field_types<ScalarField<mesh_t, value_t>> : public crtp_field<ScalarField<mesh_t, value_t>>
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
                mpi::communicator world;
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
                            std::cerr << "[" << world.rank() << "] READ NaN in " << cell << std::endl;
                            throw std::runtime_error("READ NaN");
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
                mpi::communicator world;
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
                            std::cerr << "[" << world.rank() << "] READ NaN in " << cell << std::endl;
                            throw std::runtime_error("READ NaN");
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
                this->derived_cast().m_ghosts_updated = false;
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

    // ------------------------------------------------------------------------
    // class ScalarField
    // ------------------------------------------------------------------------

    template <class mesh_t_, class value_t = double>
    class ScalarField : public field_expression<ScalarField<mesh_t_, value_t>>,
                        public inner_mesh_type<mesh_t_>,
                        public detail::inner_field_types<ScalarField<mesh_t_, value_t>>,
                        public detail::FieldBase<ScalarField<mesh_t_, value_t>>
    {
      public:

        using self_type    = ScalarField<mesh_t_, value_t>;
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

        static constexpr size_type n_comp = 1;
        static constexpr bool is_scalar   = true;
        using inner_types::static_layout;

        ScalarField() = default;

        ScalarField(std::string name, mesh_t& mesh);

        template <class E>
        ScalarField(const field_expression<E>& e);

        ScalarField(const ScalarField&);
        ScalarField& operator=(const ScalarField&);

        ScalarField(ScalarField&&) noexcept            = default;
        ScalarField& operator=(ScalarField&&) noexcept = default;

        ~ScalarField() = default;

        template <class E>
        ScalarField& operator=(const field_expression<E>& e);

        void fill(value_type v);

      private:

        std::string m_name;

        bc_container p_bc;

        bool m_ghosts_updated = false;

        friend struct detail::inner_field_types<ScalarField<mesh_t, value_t>>;
        friend class detail::FieldBase<ScalarField<mesh_t, value_t>>;
    };

    // ScalarField constructors -----------------------------------------------

    template <class mesh_t, class value_t>
    inline ScalarField<mesh_t, value_t>::ScalarField(std::string name, mesh_t& mesh)
        : inner_mesh_t(mesh)
        , inner_types()
        , m_name(std::move(name))
    {
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
        : inner_mesh_t(field.mesh())
        , inner_types(field)
        , m_name(field.m_name)
    {
        this->copy_bc_from(field);
    }

    // ScalarField operators --------------------------------------------------

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::operator=(const ScalarField& field) -> ScalarField&
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

    template <class mesh_t, class value_t>
    template <class E>
    inline auto ScalarField<mesh_t, value_t>::operator=(const field_expression<E>& e) -> ScalarField&
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

    // ScalarField methods (type-specific) ------------------------------------

    // --- fill ---------------------------------------------------------------

    template <class mesh_t, class value_t>
    inline void ScalarField<mesh_t, value_t>::fill(value_type v)
    {
        this->m_storage.data().fill(v);
        m_ghosts_updated = false;
    }

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
        return make_scalar_field<default_value_t>(name, mesh, std::forward<Func>(f));
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
