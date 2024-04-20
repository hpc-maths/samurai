// Copyright 2018-2024 the samurai's authors
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
#include "bc.hpp"
#include "cell.hpp"
#include "cell_array.hpp"
#include "field_expression.hpp"
#include "timers.hpp"
// #include "hdf5.hpp"
#include "mesh_holder.hpp"
#include "numeric/gauss_legendre.hpp"

#include "storage/eigen.hpp"

// #include "storage/xtensor.hpp"

namespace samurai
{

    template <class value_t, std::size_t size = 1, bool SOA = false>
    using field_data_storage_t = eigen_container<value_t, size, SOA>;

    template <class mesh_t, class value_t, std::size_t size = 1, bool SOA = false>
    class Field;

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

        template <class mesh_t, class value_t>
        struct inner_field_types<Field<mesh_t, value_t, 1>> : public crtp_field<Field<mesh_t, value_t, 1>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t                 = typename mesh_t::interval_t;
            using index_t                    = typename interval_t::index_t;
            using cell_t                     = Cell<dim, interval_t>;
            using data_type                  = field_data_storage_t<value_t, 1>;

            inline const value_t& operator[](index_t i) const
            {
                return m_storage.data()[static_cast<std::size_t>(i)];
            }

            inline value_t& operator[](index_t i)
            {
                return m_storage.data()[static_cast<std::size_t>(i)];
            }

            inline const value_t& operator[](const cell_t& cell) const
            {
                return m_storage.data()[static_cast<std::size_t>(cell.index)];
            }

            inline value_t& operator[](const cell_t& cell)
            {
                return m_storage.data()[static_cast<std::size_t>(cell.index)];
            }

            inline const value_t& operator()(std::size_t i) const
            {
                return m_storage.data()[i];
            }

            inline value_t& operator()(std::size_t i)
            {
                return m_storage.data()[i];
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
                        if (std::isnan(this->derived_cast().m_data[static_cast<std::size_t>(i)]))
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

            void resize()
            {
                m_storage.resize(this->derived_cast().mesh().nb_cells());
#ifdef SAMURAI_CHECK_NAN
                if constexpr (std::is_floating_point_v<value_t>)
                {
                    this->derived_cast().m_data.fill(std::nan(""));
                }
#endif
            }

            data_type m_storage;
        };

        template <class mesh_t, class value_t, std::size_t size, bool SOA>
        struct inner_field_types<Field<mesh_t, value_t, size, SOA>, std::enable_if<(size > 1)>>
            : public crtp_field<Field<mesh_t, value_t, size, SOA>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t                 = typename mesh_t::interval_t;
            using index_t                    = typename interval_t::index_t;
            using cell_t                     = Cell<dim, interval_t>;
            using data_type                  = field_data_storage_t<value_t, size, SOA>;

            inline auto operator[](index_t i) const
            {
                return view(m_storage, static_cast<std::size_t>(i));
            }

            inline auto operator[](index_t i)
            {
                return view(m_storage, static_cast<std::size_t>(i));
            }

            inline auto operator[](const cell_t& cell) const
            {
                return view(m_storage, cell.index);
            }

            inline auto operator[](const cell_t& cell)
            {
                return view(m_storage, cell.index);
            }

            inline auto operator()(std::size_t i) const
            {
                return view(m_storage, i);
            }

            inline auto operator()(std::size_t i)
            {
                return view(m_storage, i);
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
                m_storage.resize(this->derived_cast().mesh().nb_cells());
#ifdef SAMURAI_CHECK_NAN
                this->derived_cast().m_data.fill(std::nan(""));
#endif
            }

            data_type m_storage;
        };
    } // namespace detail

    template <class Field, bool is_const>
    class Field_iterator;

    template <class mesh_t_, class value_t = double, std::size_t size_, bool SOA>
    class Field : public field_expression<Field<mesh_t_, value_t, size_, SOA>>,
                  public inner_mesh_type<mesh_t_>,
                  public detail::inner_field_types<Field<mesh_t_, value_t, size_, SOA>>
    {
      public:

        static constexpr std::size_t size = size_;
        static constexpr bool is_soa      = SOA;

        using self_type    = Field<mesh_t_, value_t, size_, SOA>;
        using inner_mesh_t = inner_mesh_type<mesh_t_>;
        using mesh_t       = mesh_t_;

        using value_type  = value_t;
        using inner_types = detail::inner_field_types<Field<mesh_t, value_t, size, SOA>>;
        using data_type   = typename inner_types::data_type::container_t;
        using inner_types::operator();
        using bc_container = std::vector<std::unique_ptr<Bc<Field>>>;

        using inner_types::dim;
        using interval_t = typename mesh_t::interval_t;
        using cell_t     = typename inner_types::cell_t;

        using iterator               = Field_iterator<self_type, false>;
        using const_iterator         = Field_iterator<const self_type, true>;
        using reverse_iterator       = Field_reverse_iterator<iterator>;
        using const_reverse_iterator = Field_reverse_iterator<const_iterator>;

        Field() = default;

        Field(std::string name, mesh_t& mesh);

        template <class E>
        Field(const field_expression<E>& e);

        Field(const Field&);
        Field& operator=(const Field&);

        Field(Field&&) noexcept            = default;
        Field& operator=(Field&&) noexcept = default;

        ~Field() = default;

        template <class E>
        Field& operator=(const field_expression<E>& e);

        void fill(value_type v);

        const data_type& array() const;
        data_type& array();

        const std::string& name() const;
        std::string& name();

        void to_stream(std::ostream& os) const;

        template <class Bc_derived>
        auto attach_bc(const Bc_derived& bc);
        auto& get_bc();
        const auto& get_bc() const;
        void copy_bc_from(const Field& other);

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

        friend struct detail::inner_field_types<Field<mesh_t, value_t, size_, SOA>>;
    };

    template <class Field, bool is_const>
    class Field_iterator : public xtl::xrandom_access_iterator_base3<Field_iterator<Field, is_const>,
                                                                     CellArray_iterator<const typename Field::mesh_t::ca_type, true>>
    {
      public:

        using self_type       = Field_iterator<Field, is_const>;
        using ca_iterator     = CellArray_iterator<const typename Field::mesh_t::ca_type, true>;
        using reference       = xt::xview<typename Field::data_type&, xt::xstepped_range<long>>;
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

    template <class Field, bool is_const>
    Field_iterator<Field, is_const>::Field_iterator(Field* field, const ca_iterator& ca_it)
        : p_field(field)
        , m_ca_it(ca_it)
    {
    }

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

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline void Field<mesh_t, value_t, size_, SOA>::fill(value_type v)

    {
        this->m_storage.data().fill(v);
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline Field<mesh_t, value_t, size_, SOA>::Field(std::string name, mesh_t& mesh)
        : inner_mesh_t(mesh)
        , inner_types()
        , m_name(std::move(name))
    {
        this->resize();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    template <class E>
    inline Field<mesh_t, value_t, size_, SOA>::Field(const field_expression<E>& e)
        : inner_mesh_t(detail::extract_mesh(e.derived_cast()))
    {
        this->resize();
        *this = e;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline Field<mesh_t, value_t, size_, SOA>::Field(const Field& field)
        : inner_mesh_t(field.mesh())
        , inner_types(field)
        , m_name(field.m_name)
    {
        copy_bc_from(field);
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::operator=(const Field& field) -> Field&
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
        times::timers.stop("field expressions");
        return *this;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    template <class E>
    inline auto Field<mesh_t, value_t, size_, SOA>::operator=(const field_expression<E>& e) -> Field&
    {
        times::timers.start("field expressions");
        for_each_interval(this->mesh(),
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              (*this)(level, i, index) = e.derived_cast()(level, i, index);
                          });
        times::timers.stop("field expressions");
        return *this;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    template <class... T>
    inline auto Field<mesh_t, value_t, size_, SOA>::get_interval(std::size_t level,
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

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto
    Field<mesh_t, value_t, size_, SOA>::get_interval(std::size_t level,
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

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::array() const -> const data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::array() -> data_type&
    {
        return this->m_storage.data();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline const std::string& Field<mesh_t, value_t, size_, SOA>::name() const
    {
        return m_name;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline std::string& Field<mesh_t, value_t, size_, SOA>::name()
    {
        return m_name;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline void Field<mesh_t, value_t, size_, SOA>::to_stream(std::ostream& os) const
    {
        os << "Field " << m_name << "\n";

#ifdef SAMURAI_CHECK_NAN
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
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

    template <class mesh_t, class T, std::size_t N, bool SOA>
    inline std::ostream& operator<<(std::ostream& out, const Field<mesh_t, T, N, SOA>& field)
    {
        field.to_stream(out);
        return out;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    template <class Bc_derived>
    inline auto Field<mesh_t, value_t, size_, SOA>::attach_bc(const Bc_derived& bc)
    {
        p_bc.push_back(bc.clone());
        return p_bc.back().get();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto& Field<mesh_t, value_t, size_, SOA>::get_bc()
    {
        return p_bc;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline const auto& Field<mesh_t, value_t, size_, SOA>::get_bc() const
    {
        return p_bc;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    void Field<mesh_t, value_t, size_, SOA>::copy_bc_from(const Field<mesh_t, value_t, size_, SOA>& other)
    {
        std::transform(other.get_bc().cbegin(),
                       other.get_bc().cend(),
                       std::back_inserter(p_bc),
                       [](const auto& v)
                       {
                           return v->clone();
                       });
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::begin() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].begin());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::end() -> iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return iterator(this, this->mesh()[mesh_id_t::cells].end());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::end() const -> const_iterator
    {
        return cend();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::cbegin() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cbegin());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::cend() const -> const_iterator
    {
        using mesh_id_t = typename mesh_t::mesh_id_t;
        return const_iterator(this, this->mesh()[mesh_id_t::cells].cend());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rbegin() const -> const_reverse_iterator
    {
        return rcbegin();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rend() const -> const_reverse_iterator
    {
        return rcend();
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rcbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::rcend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    template <class value_t, std::size_t size, bool SOA = false, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh)
    {
        using field_t = Field<mesh_t, value_t, size, SOA>;
        field_t f(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        if constexpr (std::is_floating_point_v<value_t>)
        {
            f.fill(static_cast<value_t>(std::nan("")));
        }
#endif
        return f;
    }

    template <class value_t, std::size_t size, bool SOA = false, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh, value_t init_value)
    {
        using field_t = Field<mesh_t, value_t, size, SOA>;
        auto field    = field_t(name, mesh);
        field.fill(init_value);
        return field;
    }

    template <std::size_t size, bool SOA = false, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh)
    {
        using default_value_t = double;
        return make_field<default_value_t, size, SOA>(name, mesh);
    }

    template <std::size_t size, bool SOA = false, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh, double init_value)
    {
        using default_value_t = double;
        return make_field<default_value_t, size, SOA>(name, mesh, init_value);
    }

    /**
     * @brief Creates a field.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param gl Gauss Legendre polynomial
     */
    template <class value_t, std::size_t size, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_field(std::string name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        auto field = make_field<value_t, size, SOA, mesh_t>(name, mesh);
#ifdef SAMURAI_CHECK_NAN
        f.fill(std::nan(""));
#else
        field.fill(0);
#endif

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const double& h = cell.length;
                          field[cell]     = gl.template quadrature<size>(cell, f) / pow(h, mesh_t::dim);
                      });
        return field;
    }

    template <std::size_t size, bool SOA = false, class mesh_t, class Func, std::size_t polynomial_degree>
    auto make_field(std::string name, mesh_t& mesh, Func&& f, const GaussLegendre<polynomial_degree>& gl)
    {
        using default_value_t = double;
        return make_field<default_value_t, size, SOA>(name, mesh, std::forward<Func>(f), gl);
    }

    /**
     * @brief Creates a field.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     */
    template <class value_t,
              std::size_t size,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    auto make_field(std::string name, mesh_t& mesh, Func&& f)
    {
        auto field = make_field<value_t, size, SOA, mesh_t>(name, mesh);
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

    template <std::size_t size,
              bool SOA = false,
              class mesh_t,
              class Func,
              typename = std::enable_if_t<std::is_invocable_v<Func, typename Cell<mesh_t::dim, typename mesh_t::interval_t>::coords_t>>>
    auto make_field(std::string name, mesh_t& mesh, Func&& f)
    {
        using default_value_t = double;
        return make_field<default_value_t, size, SOA>(name, mesh, std::forward<Func>(f));
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline bool operator==(const Field<mesh_t, value_t, size_, SOA>& field1, const Field<mesh_t, value_t, size_, SOA>& field2)
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
                          if (std::abs(field1[cell] - field2[cell]) > 1e-15)
                          {
                              is_same = false;
                          }
                      });

        return is_same;
    }

    template <class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline bool operator!=(const Field<mesh_t, value_t, size_, SOA>& field1, const Field<mesh_t, value_t, size_, SOA>& field2)
    {
        return !(field1 == field2);
    }

    template <class TField, class... TFields>
    class Field_tuple
    {
      public:

        using tuple_type                   = std::tuple<TField&, TFields&...>;
        using tuple_type_without_ref       = std::tuple<TField, TFields...>;
        static constexpr std::size_t nelem = detail::compute_size<TField, TFields...>();
        using common_t                     = detail::common_type_t<TField, TFields...>;
        using mesh_t                       = typename TField::mesh_t;
        using mesh_id_t                    = typename mesh_t::mesh_id_t;

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
} // namespace samurai
