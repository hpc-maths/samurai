// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <filesystem>
namespace fs = std::filesystem;

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "algorithm.hpp"
#include "bc/bc.hpp"
#include "cell.hpp"
#include "cell_array.hpp"
#include "concepts.hpp"
#include "field_expression.hpp"
#include "mesh_holder.hpp"
#include "numeric/gauss_legendre.hpp"
#include "storage/containers.hpp"
#include "timers.hpp"

namespace samurai
{
    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    class VectorField;

    template <class mesh_t, class value_t>
    class ScalarField;

    template <class Field, bool is_const>
    class Field_iterator;

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

    namespace detail
    {
        template <class Field, class = void>
        struct inner_field_types;

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

        // ------------------------------------------------------------------------
        // struct inner_field_types
        // ------------------------------------------------------------------------

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
                this->derived_cast().m_ghosts_updated = false;
#ifdef SAMURAI_CHECK_NAN
                m_storage.data().fill(std::nan(""));
#endif
            }

            data_type m_storage;
        };
    } // namespace detail

    template <class Field, bool is_const>
    class Field_iterator;

    // ------------------------------------------------------------------------
    // class VectorField
    // ------------------------------------------------------------------------

    template <class mesh_t_, class value_t, std::size_t n_comp_, bool SOA>
    class VectorField : public field_expression<VectorField<mesh_t_, value_t, n_comp_, SOA>>,
                        public inner_mesh_type<mesh_t_>,
                        public detail::inner_field_types<VectorField<mesh_t_, value_t, n_comp_, SOA>>
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

        const data_type& array() const;
        data_type& array();

        const std::string& name() const;
        std::string& name();

        void to_stream(std::ostream& os) const;

        bool& ghosts_updated();
        bool ghosts_updated() const;

        template <class Bc_derived>
        auto attach_bc(const Bc_derived& bc);
        auto& get_bc();
        const auto& get_bc() const;
        void copy_bc_from(const VectorField& other);

        iterator begin();
        const_iterator begin() const;
        const_iterator cbegin() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;
        const_reverse_iterator rcbegin() const;

        iterator end();
        const_iterator end() const;
        const_iterator cend() const;

        reverse_iterator rend();
        const_reverse_iterator rend() const;
        const_reverse_iterator rcend() const;

      private:

        template <class... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const T... index) const;

        const interval_t&
        get_interval(std::size_t level, const interval_t& interval, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const;

        std::string m_name;

        bc_container p_bc;
        bool m_ghosts_updated = false;

        friend struct detail::inner_field_types<VectorField<mesh_t, value_t, n_comp_, SOA>>;
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
        copy_bc_from(field);
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

    // VectorField methods ----------------------------------------------------

    // --- element access -----------------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    template <class... T>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::get_interval(std::size_t level,
                                                                         const interval_t& interval,
                                                                         const T... index) const -> const interval_t&
    {
        const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index...);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
        {
            // using mesh_id_t  = typename mesh_t::mesh_id_t;
            // auto coords      = make_field<int, dim, false>("coordinates", this->mesh());
            // auto level_field = make_field<std::size_t, 1, false>("level", this->mesh());
            // for_each_cell(this->mesh()[mesh_id_t::reference],
            //               [&](auto& cell)
            //               {
            //                   if constexpr (dim == 1)
            //                   {
            //                       coords[cell] = cell.indices[0];
            //                   }
            //                   else
            //                   {
            //                       coords[cell] = cell.indices;
            //                   }
            //                   level_field[cell] = cell.level;
            //               });
            // save(fs::current_path(), "mesh_throw", {true, true}, this->mesh(), coords, level_field);
            (std::cout << ... << index) << std::endl;
            throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
        }

        return interval_tmp;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::get_interval(std::size_t level,
                                                                         const interval_t& interval,
                                                                         const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const
        -> const interval_t&
    {
        const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
        {
            throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
        }

        return interval_tmp;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::array() const -> const data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::array() -> data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline const std::string& VectorField<mesh_t, value_t, n_comp_, SOA>::name() const
    {
        return m_name;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline std::string& VectorField<mesh_t, value_t, n_comp_, SOA>::name()
    {
        return m_name;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline bool& VectorField<mesh_t, value_t, n_comp_, SOA>::ghosts_updated()
    {
        return m_ghosts_updated;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline bool VectorField<mesh_t, value_t, n_comp_, SOA>::ghosts_updated() const
    {
        return m_ghosts_updated;
    }

    // --- iterators ----------------------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::begin() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].cbegin());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::end() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].cend());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::end() const -> const_iterator
    {
        return cend();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::cbegin() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cbegin());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::cend() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cend());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    // --- others -------------------------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline void VectorField<mesh_t, value_t, n_comp_, SOA>::fill(value_type v)
    {
        this->m_storage.data().fill(v);
        m_ghosts_updated = false;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline void VectorField<mesh_t, value_t, n_comp_, SOA>::to_stream(std::ostream& os) const
    {
        os << "Field " << m_name << "\n";

#ifdef SAMURAI_CHECK_NAN
        using mesh_id_t = typename VectorField::mesh_t::mesh_id_t;
        for_each_cell(this->mesh()[mesh_id_t::reference],
#else
        for_each_cell(this->mesh(),
#endif
                      [&](auto& cell)
                      {
                          os << "\tlevel: " << cell.level << " coords: " << cell.center() << " index: " << cell.index
                             << ", value: " << this->operator[](cell) << "\n";
                      });
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    template <class Bc_derived>
    inline auto VectorField<mesh_t, value_t, n_comp_, SOA>::attach_bc(const Bc_derived& bc)
    {
        if (bc.stencil_size() > this->mesh().cfg().max_stencil_size())
        {
            std::cerr << "The stencil size required by this boundary condition (" << bc.stencil_size()
                      << ") is larger than the max_stencil_size parameter of the mesh (" << this->mesh().cfg().max_stencil_size()
                      << ").\nYou can set it with mesh_config.max_stencil_radius(" << bc.stencil_size() / 2
                      << ") or mesh_config.max_stencil_size(" << bc.stencil_size() << ")." << std::endl;
            exit(EXIT_FAILURE);
        }
        p_bc.push_back(bc.clone());
        return p_bc.back().get();
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline auto& VectorField<mesh_t, value_t, n_comp_, SOA>::get_bc()
    {
        return p_bc;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline const auto& VectorField<mesh_t, value_t, n_comp_, SOA>::get_bc() const
    {
        return p_bc;
    }

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    void VectorField<mesh_t, value_t, n_comp_, SOA>::copy_bc_from(const VectorField<mesh_t, value_t, n_comp_, SOA>& other)
    {
        std::transform(other.get_bc().cbegin(),
                       other.get_bc().cend(),
                       std::back_inserter(p_bc),
                       [](const auto& v)
                       {
                           return v->clone();
                       });
    }

    // VectorField extern operators -------------------------------------------

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline std::ostream& operator<<(std::ostream& out, const VectorField<mesh_t, value_t, n_comp_, SOA>& field)
    {
        field.to_stream(out);
        return out;
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

    template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA>
    inline bool operator!=(const VectorField<mesh_t, value_t, n_comp_, SOA>& field1, const VectorField<mesh_t, value_t, n_comp_, SOA>& field2)
    {
        return !(field1 == field2);
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

    // ------------------------------------------------------------------------
    // class ScalarField
    // ------------------------------------------------------------------------

    template <class mesh_t_, class value_t = double>
    class ScalarField : public field_expression<ScalarField<mesh_t_, value_t>>,
                        public inner_mesh_type<mesh_t_>,
                        public detail::inner_field_types<ScalarField<mesh_t_, value_t>>
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

        const data_type& array() const;
        data_type& array();

        const std::string& name() const;
        std::string& name();

        void to_stream(std::ostream& os) const;

        bool& ghosts_updated();
        bool ghosts_updated() const;

        template <class Bc_derived>
        auto attach_bc(const Bc_derived& bc);
        auto& get_bc();
        const auto& get_bc() const;
        void copy_bc_from(const ScalarField& other);

        iterator begin();
        const_iterator begin() const;
        const_iterator cbegin() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;
        const_reverse_iterator rcbegin() const;

        iterator end();
        const_iterator end() const;
        const_iterator cend() const;

        reverse_iterator rend();
        const_reverse_iterator rend() const;
        const_reverse_iterator rcend() const;

      private:

        template <class... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const T... index) const;

        const interval_t&
        get_interval(std::size_t level, const interval_t& interval, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const;

        std::string m_name;

        bc_container p_bc;

        bool m_ghosts_updated = false;

        friend struct detail::inner_field_types<ScalarField<mesh_t, value_t>>;
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
        copy_bc_from(field);
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

    // ScalarField methods ----------------------------------------------------

    // --- element access -----------------------------------------------------

    template <class mesh_t, class value_t>
    template <class... T>
    inline auto
    ScalarField<mesh_t, value_t>::get_interval(std::size_t level, const interval_t& interval, const T... index) const -> const interval_t&
    {
        const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index...);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
        {
            // using mesh_id_t  = typename mesh_t::mesh_id_t;
            // auto coords      = make_field<int, dim, false>("coordinates", this->mesh());
            // auto level_field = make_field<std::size_t, 1, false>("level", this->mesh());
            // for_each_cell(this->mesh()[mesh_id_t::reference],
            //               [&](auto& cell)
            //               {
            //                   if constexpr (dim == 1)
            //                   {
            //                       coords[cell] = cell.indices[0];
            //                   }
            //                   else
            //                   {
            //                       coords[cell] = cell.indices;
            //                   }
            //                   level_field[cell] = cell.level;
            //               });
            // save(fs::current_path(), "mesh_throw", {true, true}, this->mesh(), coords, level_field);
            (std::cout << ... << index) << std::endl;
            throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
        }

        return interval_tmp;
    }

    template <class mesh_t, class value_t>
    inline auto
    ScalarField<mesh_t, value_t>::get_interval(std::size_t level,
                                               const interval_t& interval,
                                               const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const -> const interval_t&
    {
        const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) || (interval_tmp.start > interval.start))
        {
            throw std::out_of_range(fmt::format("FIELD ERROR on level {}: try to find interval {}", level, interval));
        }

        return interval_tmp;
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::array() const -> const data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::array() -> data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t>
    inline const std::string& ScalarField<mesh_t, value_t>::name() const
    {
        return m_name;
    }

    template <class mesh_t, class value_t>
    inline std::string& ScalarField<mesh_t, value_t>::name()
    {
        return m_name;
    }

    template <class mesh_t, class value_t>
    inline bool& ScalarField<mesh_t, value_t>::ghosts_updated()
    {
        return m_ghosts_updated;
    }

    template <class mesh_t, class value_t>
    inline bool ScalarField<mesh_t, value_t>::ghosts_updated() const
    {
        return m_ghosts_updated;
    }

    // --- iterators ----------------------------------------------------------

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::begin() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].cbegin());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::end() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].cend());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::end() const -> const_iterator
    {
        return cend();
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::cbegin() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cbegin());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::cend() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cend());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class mesh_t, class value_t>
    inline auto ScalarField<mesh_t, value_t>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    // --- others -------------------------------------------------------------

    template <class mesh_t, class value_t>
    inline void ScalarField<mesh_t, value_t>::fill(value_type v)
    {
        this->m_storage.data().fill(v);
    }

    template <class mesh_t, class value_t>
    inline void ScalarField<mesh_t, value_t>::to_stream(std::ostream& os) const
    {
        os << "Field " << m_name << "\n";

#ifdef SAMURAI_CHECK_NAN
        using mesh_id_t = typename ScalarField::mesh_t::mesh_id_t;
        for_each_cell(this->mesh()[mesh_id_t::reference],
#else
        for_each_cell(this->mesh(),
#endif
                      [&](auto& cell)
                      {
                          os << "\tlevel: " << cell.level << " coords: " << cell.center() << " index: " << cell.index
                             << ", value: " << this->operator[](cell) << "\n";
                      });
    }

    template <class mesh_t, class value_t>
    template <class Bc_derived>
    inline auto ScalarField<mesh_t, value_t>::attach_bc(const Bc_derived& bc)
    {
        if (bc.stencil_size() > this->mesh().cfg().max_stencil_size())
        {
            std::cerr << "The stencil size required by this boundary condition (" << bc.stencil_size()
                      << ") is larger than the max_stencil_size parameter of the mesh (" << this->mesh().cfg().max_stencil_size()
                      << ").\nYou can set it with mesh_config.max_stencil_radius(" << bc.stencil_size() / 2
                      << ") or mesh_config.max_stencil_size(" << bc.stencil_size() << ")." << std::endl;
            exit(EXIT_FAILURE);
        }
        p_bc.push_back(bc.clone());
        return p_bc.back().get();
    }

    template <class mesh_t, class value_t>
    inline auto& ScalarField<mesh_t, value_t>::get_bc()
    {
        return p_bc;
    }

    template <class mesh_t, class value_t>
    inline const auto& ScalarField<mesh_t, value_t>::get_bc() const
    {
        return p_bc;
    }

    template <class mesh_t, class value_t>
    void ScalarField<mesh_t, value_t>::copy_bc_from(const ScalarField<mesh_t, value_t>& other)
    {
        std::transform(other.get_bc().cbegin(),
                       other.get_bc().cend(),
                       std::back_inserter(p_bc),
                       [](const auto& v)
                       {
                           return v->clone();
                       });
    }

    // ScalarField extern operators -------------------------------------------

    template <class mesh_t, class value_t>
    inline std::ostream& operator<<(std::ostream& out, const ScalarField<mesh_t, value_t>& field)
    {
        field.to_stream(out);
        return out;
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

    // ------------------------------------------------------------------------
    // class Field_tuple
    // ------------------------------------------------------------------------

    template <class TField, class... TFields>
    class Field_tuple
    {
      public:

        using tuple_type             = std::tuple<TField&, TFields&...>;
        using tuple_type_without_ref = std::tuple<TField, TFields...>;
        using common_t               = detail::common_type_t<TField, TFields...>;
        using mesh_t                 = typename TField::mesh_t;
        using mesh_id_t              = typename mesh_t::mesh_id_t;
        using size_type              = typename TField::size_type;

        Field_tuple(TField& field, TFields&... fields)
            : m_fields(field, fields...)
        {
        }

        const auto& mesh() const
        {
            return std::get<0>(m_fields).mesh();
        }

        auto& mesh()
        {
            return std::get<0>(m_fields).mesh();
        }

        const auto& elements() const
        {
            return m_fields;
        }

        auto& elements()
        {
            return m_fields;
        }

      private:

        tuple_type m_fields;
    };

    template <class Field1, class Field2>
        requires(IsField<Field1> && IsField<Field2>)
    inline void swap(Field1& u1, Field2& u2)
    {
        std::swap(u1.array(), u2.array());

        bool u1_ghosts_updated = u1.ghosts_updated();
        u1.ghosts_updated()    = u2.ghosts_updated();
        u2.ghosts_updated()    = u1_ghosts_updated;
    }
} // namespace samurai
