// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <memory>

#include <spdlog/spdlog.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "algorithm.hpp"
#include "cell.hpp"
#include "field_expression.hpp"
// #include "mr/mesh.hpp"
#include "mr/operators.hpp"

namespace samurai
{
    template<class mesh_t, class value_t, std::size_t size>
    class Field;

    namespace detail
    {
        template<class Field>
        struct inner_field_types;

        template <class D>
        struct crtp_field
        {
            using derived_type = D;

            derived_type &derived_cast() & noexcept
            {
                return *static_cast<derived_type *>(this);
            }

            const derived_type &derived_cast() const &noexcept
            {
                return *static_cast<const derived_type *>(this);
            }

            derived_type derived_cast() && noexcept
            {
                return *static_cast<derived_type *>(this);
            }
        };

        template<class mesh_t, class value_t>
        struct inner_field_types<Field<mesh_t, value_t, 1>>: public crtp_field<Field<mesh_t, value_t, 1>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t = typename mesh_t::interval_t;
            using cell_t = Cell<typename interval_t::coord_index_t, dim>;
            using data_type = xt::xtensor<value_t, 1>;

            inline const value_t& operator[](std::size_t i) const
            {
                return this->derived_cast().m_data[i];
            }

            inline value_t& operator[](std::size_t i)
            {
                return this->derived_cast().m_data[i];
            }

            inline const value_t& operator[](const cell_t& cell) const
            {
                return this->derived_cast().m_data[cell.index];
            }

            inline value_t& operator[](const cell_t& cell)
            {
                return this->derived_cast().m_data[cell.index];
            }

            inline const value_t& operator()(std::size_t i) const
            {
                return this->derived_cast().m_data[i];
            }

            inline value_t& operator()(std::size_t i)
            {
                return this->derived_cast().m_data[i];
            }

            void resize()
            {
                this->derived_cast().m_data.resize({this->derived_cast().p_mesh->nb_cells()});
            }
        };

        template<class mesh_t, class value_t, std::size_t size>
        struct inner_field_types<Field<mesh_t, value_t, size>>: public crtp_field<Field<mesh_t, value_t, size>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t = typename mesh_t::interval_t;
            using cell_t = Cell<typename interval_t::coord_index_t, dim>;
            using data_type = xt::xtensor<value_t, 2>;

            inline auto operator[](std::size_t i) const
            {
                return xt::view(this->derived_cast().m_data, i);
            }

            inline auto operator[](std::size_t i)
            {
                return xt::view(this->derived_cast().m_data, i);
            }

            inline auto operator[](const cell_t& cell) const
            {
                return xt::view(this->derived_cast().m_data, cell.index);
            }

            inline auto operator[](const cell_t& cell)
            {
                return xt::view(this->derived_cast().m_data, cell.index);
            }

            inline auto operator()(std::size_t i) const
            {
                return xt::view(this->derived_cast().m_data, i);
            }

            inline auto operator()(std::size_t i)
            {
                return xt::view(this->derived_cast().m_data, i);
            }

            template<class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step), item);
            }

            template<class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t &interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step), item);
            }

            template<class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step), xt::range(item_s, item_e));
            }

            template<class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step), xt::range(item_s, item_e));
            }

            void resize()
            {
                this->derived_cast().m_data.resize({this->derived_cast().p_mesh->nb_cells(), size});
            }

        };

    } // namespace detail

    template<class mesh_t_, class value_t = double, std::size_t size_ = 1>
    class Field : public field_expression<Field<mesh_t_, value_t, size_>>,
                  public detail::inner_field_types<Field<mesh_t_, value_t, size_>>
    {
      public:
        static constexpr std::size_t size = size_;

        using mesh_t = mesh_t_;

        using value_type = value_t;
        using inner_types = detail::inner_field_types<Field<mesh_t, value_t, size>>;
        using data_type = typename inner_types::data_type;
        using inner_types::operator();

        using inner_types::dim;
        using interval_t = typename mesh_t::interval_t;
        using cell_t = Cell<typename interval_t::coord_index_t, dim>;


        Field(const std::string& name, mesh_t& mesh);

        Field() = default;

        Field(const Field&) = default;
        Field& operator=(const Field&) = default;

        Field(Field&&) = default;
        Field& operator=(Field&&) = default;

        template<class E>
        Field &operator=(const field_expression<E> &e);

        template<class... T>
        auto operator()(const std::size_t level, const interval_t& interval, const T... index);

        template<class... T>
        auto operator()(const std::size_t level, const interval_t& interval, const T... index) const;

        void fill(value_type v);

        const data_type& array() const;
        data_type& array();

        std::string name() const;

        const mesh_t& mesh() const;
        mesh_t& mesh();

        const mesh_t* mesh_ptr() const;
        mesh_t* mesh_ptr();

        void to_stream(std::ostream& os) const;

    private:
        template<class... T>
        const interval_t& get_interval(std::string rw, const std::size_t level, const interval_t &interval, const T... index) const;

        std::string m_name;
        mesh_t* p_mesh;
        data_type m_data;

        friend class detail::inner_field_types<Field<mesh_t, value_t, size_>>;
    };

    template<class mesh_t, class value_t, std::size_t size_>
    inline void Field<mesh_t, value_t, size_>::fill(value_type v)
    {
        m_data.fill(v);
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline Field<mesh_t, value_t, size_>::Field(const std::string& name, mesh_t& mesh)
        : m_name(name), p_mesh(&mesh)
    {
        this->resize();
    }

    template<class mesh_t, class value_t, std::size_t size_>
    template<class E>
    inline auto Field<mesh_t, value_t, size_>::operator=(const field_expression<E> &e) -> Field&
    {
        // FIX: this works only when the mesh_t is a derived class of Mesh_base.
        //      CellArray has no type mesh_id_t.
        //
        using mesh_id_t = typename mesh_t::mesh_id_t;

        auto min_level = (*p_mesh)[mesh_id_t::cells].min_level();
        auto max_level = (*p_mesh)[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection((*p_mesh)[mesh_id_t::cells][level],
                                       (*p_mesh)[mesh_id_t::cells][level]);

            subset.apply_op(apply_expr(*this, e));
        }
        return *this;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    template<class... T>
    inline auto Field<mesh_t, value_t, size_>::get_interval(std::string rw, const std::size_t level,
                                                            const interval_t &interval, const T... index) const -> const interval_t&
    {
        const interval_t& interval_tmp = p_mesh->get_interval(level, interval, index...);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
            (interval_tmp.start > interval.start))
        {
            spdlog::critical("{} FIELD ERROR on level {}: try to find interval {}",
                             rw, level, interval);
            throw;
        }

        return interval_tmp;
    }


    template<class mesh_t, class value_t, std::size_t size_>
    template<class... T>
    inline auto Field<mesh_t, value_t, size_>::operator()(const std::size_t level,
                                                          const interval_t &interval, const T... index)
    {
        auto interval_tmp = get_interval("READ OR WRITE", level, interval, index...);
        return xt::view(m_data,
                        xt::range(interval_tmp.index + interval.start,
                                    interval_tmp.index + interval.end,
                                    interval.step));
    }

    template<class mesh_t, class value_t, std::size_t size_>
    template<class... T>
    inline auto Field<mesh_t, value_t, size_>::operator()(const std::size_t level,
                                                          const interval_t &interval, const T... index) const
    {
        auto interval_tmp = get_interval("READ", level, interval, index...);
        return xt::view(m_data,
                        xt::range(interval_tmp.index + interval.start,
                                    interval_tmp.index + interval.end,
                                    interval.step));
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::array() const -> const data_type&
    {
        return m_data;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::array() -> data_type&
    {
        return m_data;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline std::string Field<mesh_t, value_t, size_>::name() const
    {
        return m_name;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::mesh() const -> const mesh_t&
    {
        return *p_mesh;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::mesh() -> mesh_t&
    {
        return *p_mesh;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::mesh_ptr() const -> const mesh_t*
    {
        return p_mesh;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline auto Field<mesh_t, value_t, size_>::mesh_ptr() -> mesh_t*
    {
        return p_mesh;
    }

    template<class mesh_t, class value_t, std::size_t size_>
    inline void Field<mesh_t, value_t, size_>::to_stream(std::ostream& os) const
    {
        os << "Field " << m_name << "\n";
        for_each_cell(*p_mesh, [&](auto &cell)
        {
                os << "\tlevel: " << cell.level << " coords: " << cell.center()
                    << " value: " << xt::view(m_data, cell.index) << "\n";
        });
    }

    template<class mesh_t, class T, std::size_t N>
    inline std::ostream& operator<<(std::ostream& out, const Field<mesh_t, T, N>& field)
    {
        field.to_stream(out);
        return out;
    }

    template <class value_t, std::size_t size, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh)
    {
        using field_t = Field<mesh_t, value_t, size>;
        return field_t(name, mesh);
    }

} // namespace samurai