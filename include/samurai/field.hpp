// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <memory>

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "algorithm.hpp"
#include "cell.hpp"
#include "field_expression.hpp"
#include "mr/operators.hpp"
#include "numeric/gauss_legendre.hpp"
#include "boundary_condition.hpp"

namespace samurai
{
    template<class mesh_t, class value_t, std::size_t size, bool SOA>
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
        struct inner_field_types<Field<mesh_t, value_t, 1, false>>: public crtp_field<Field<mesh_t, value_t, 1, false>>
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

            template<class... T>
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("READ OR WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
            }

            template<class... T>
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
            }

            void resize()
            {
                this->derived_cast().m_data.resize({this->derived_cast().mesh().nb_cells()});
            }
        };

        template<class mesh_t, class value_t>
        struct inner_field_types<Field<mesh_t, value_t, 1, true>>: public inner_field_types<Field<mesh_t, value_t, 1, false>>
        {};

        template<class mesh_t, class value_t, std::size_t size>
        struct inner_field_types<Field<mesh_t, value_t, size, false>>: public crtp_field<Field<mesh_t, value_t, size, false>>
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
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("READ OR WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
            }

            template<class... T>
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data,
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
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
                this->derived_cast().m_data.resize({this->derived_cast().mesh().nb_cells(), size});
            }

        };

        template<class mesh_t, class value_t, std::size_t size>
        struct inner_field_types<Field<mesh_t, value_t, size, true>>: public crtp_field<Field<mesh_t, value_t, size, true>>
        {
            static constexpr std::size_t dim = mesh_t::dim;
            using interval_t = typename mesh_t::interval_t;
            using cell_t = Cell<typename interval_t::coord_index_t, dim>;
            using data_type = xt::xtensor<value_t, 2>;

            inline auto operator[](std::size_t i) const
            {
                return xt::view(this->derived_cast().m_data, xt::all(), i);
            }

            inline auto operator[](std::size_t i)
            {
                return xt::view(this->derived_cast().m_data, xt::all(), i);
            }

            inline auto operator[](const cell_t& cell) const
            {
                return xt::view(this->derived_cast().m_data, xt::all(), cell.index);
            }

            inline auto operator[](const cell_t& cell)
            {
                return xt::view(this->derived_cast().m_data, xt::all(), cell.index);
            }

            inline auto operator()(std::size_t i) const
            {
                return xt::view(this->derived_cast().m_data, xt::all(), i);
            }

            inline auto operator()(std::size_t i)
            {
                return xt::view(this->derived_cast().m_data, xt::all(), i);
            }

            template<class... T>
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("READ OR WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data, xt::all(),
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
            }

            template<class... T>
            inline auto operator()(const std::size_t level, const interval_t &interval, const T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data, xt::all(),
                                xt::range(interval_tmp.index + interval.start,
                                          interval_tmp.index + interval.end,
                                          interval.step));
            }

            template<class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data, item,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step));
            }

            template<class... T>
            inline auto operator()(std::size_t item, std::size_t level, const interval_t &interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data, item,
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step));
            }

            template<class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval("WRITE", level, interval, index...);
                return xt::view(this->derived_cast().m_data, xt::range(item_s, item_e),
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step));
            }

            template<class... T>
            inline auto operator()(std::size_t item_s, std::size_t item_e, std::size_t level, const interval_t& interval, T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval("READ", level, interval, index...);
                return xt::view(this->derived_cast().m_data, xt::range(item_s, item_e),
                                xt::range(interval_tmp.index + interval.start,
                                            interval_tmp.index + interval.end,
                                            interval.step));
            }

            void resize()
            {
                this->derived_cast().m_data.resize({size, this->derived_cast().p_mesh->nb_cells()});
            }

        };

    } // namespace detail

    template <class Mesh>
    class hold
    {
        public:
            static constexpr std::size_t dim = Mesh::dim;
            using interval_t = typename Mesh::interval_t;

            hold(Mesh& mesh)
            : m_mesh(mesh)
            {}

            Mesh& get()
            {
                return m_mesh;
            }

        private:
            Mesh& m_mesh;
    };

    template <class Mesh>
    auto holder(Mesh& mesh)
    {
        return hold<Mesh>(mesh);
    }

    template<class Mesh>
    class inner_mesh_type
    {
        public:
            using mesh_t = Mesh;

            inner_mesh_type(mesh_t& mesh)
            : p_mesh(&mesh)
            {}

            const mesh_t& mesh() const
            {
                return *p_mesh;
            }
            mesh_t& mesh()
            {
                return *p_mesh;
            }

            const mesh_t* mesh_ptr() const
            {
                return p_mesh;
            }
            mesh_t* mesh_ptr()
            {
                return p_mesh;
            }

        private:
            mesh_t* p_mesh;
    };

    template<class Mesh>
    class inner_mesh_type<hold<Mesh>>
    {
        public:
            using mesh_t = Mesh;

            inner_mesh_type(hold<Mesh>& mesh)
            : m_mesh(mesh.get())
            {}

            const mesh_t& mesh() const
            {
                return m_mesh;
            }
            mesh_t& mesh()
            {
                return m_mesh;
            }

            const mesh_t* mesh_ptr() const
            {
                return &m_mesh;
            }
            mesh_t* mesh_ptr()
            {
                return &m_mesh;
            }

        private:
            mesh_t m_mesh;
    };

    template<class mesh_t_, class value_t = double, std::size_t size_ = 1, bool SOA=false>
    class Field : public field_expression<Field<mesh_t_, value_t, size_, SOA>>,
                  public detail::inner_field_types<Field<mesh_t_, value_t, size_, SOA>>,
                  public inner_mesh_type<mesh_t_>
    {
      public:
        static constexpr std::size_t size = size_;
        static constexpr bool is_soa = SOA;

        using inner_mesh_t = inner_mesh_type<mesh_t_>;
        using mesh_t = mesh_t_;

        using value_type = value_t;
        using inner_types = detail::inner_field_types<Field<mesh_t, value_t, size, SOA>>;
        using data_type = typename inner_types::data_type;
        using inner_types::operator();

        using inner_types::dim;
        using interval_t = typename mesh_t::interval_t;
        using cell_t = Cell<typename interval_t::coord_index_t, dim>;

        using boundary_condition_t = BoundaryCondition<value_t, dim>;
        using boundary_point_t = typename boundary_condition_t::boundary_point_t;
        using boundary_part_t = typename boundary_condition_t::boundary_part_t;
        using boundary_cond_t = typename boundary_condition_t::boundary_cond_t;

      private:
        std::vector<boundary_condition_t> m_boundary_conditions;

      public:
        Field(const std::string& name, mesh_t& mesh);

        Field() = default;

        Field(const Field&) = default;
        Field& operator=(const Field&) = default;

        Field(Field&&) = default;
        Field& operator=(Field&&) = default;

        template<class E>
        Field &operator=(const field_expression<E> &e);

        // template<class... T>
        // auto operator()(const std::size_t level, const interval_t& interval, const T... index);

        // template<class... T>
        // auto operator()(const std::size_t level, const interval_t& interval, const T... index) const;

        void fill(value_type v);

        const data_type& array() const;
        data_type& array();

        std::string name() const;

        void to_stream(std::ostream& os) const;

        const std::vector<boundary_condition_t>& boundary_conditions() const;
        boundary_condition_t& set_dirichlet(boundary_cond_t dirichlet_value);
        boundary_condition_t& set_neumann(boundary_cond_t neumann_value);

    private:
        template<class... T>
        const interval_t& get_interval(std::string rw, const std::size_t level, const interval_t &interval, const T... index) const;

        std::string m_name;
        data_type m_data;

        friend struct detail::inner_field_types<Field<mesh_t, value_t, size_, SOA>>;
    };

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline void Field<mesh_t, value_t, size_, SOA>::fill(value_type v)
    {
        m_data.fill(v);
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline Field<mesh_t, value_t, size_, SOA>::Field(const std::string& name, mesh_t& mesh)
        : inner_mesh_t(mesh), m_name(name)
    {
        this->resize();
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    template<class E>
    inline auto Field<mesh_t, value_t, size_, SOA>::operator=(const field_expression<E> &e) -> Field&
    {
        // FIX: this works only when the mesh_t is a derived class of Mesh_base.
        //      CellArray has no type mesh_id_t.
        //
        using mesh_id_t = typename mesh_t::mesh_id_t;

        auto min_level = this->mesh()[mesh_id_t::cells].min_level();
        auto max_level = this->mesh()[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(this->mesh()[mesh_id_t::cells][level],
                                       this->mesh()[mesh_id_t::cells][level]);

            subset.apply_op(apply_expr(*this, e));
        }
        return *this;
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    template<class... T>
    inline auto Field<mesh_t, value_t, size_, SOA>::get_interval(std::string rw, const std::size_t level,
                                                            const interval_t &interval, const T... index) const -> const interval_t&
    {
        const interval_t& interval_tmp = this->mesh().get_interval(level, interval, index...);

        if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) ||
            (interval_tmp.start > interval.start))
        {
            std::cout << fmt::format("{} FIELD ERROR on level {}: try to find interval {}",
                                rw, level, interval) << std::endl;
            // std::cout << this->mesh() << std::endl;
            throw;
        }

        return interval_tmp;
    }


    // template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    // template<class... T>
    // inline auto Field<mesh_t, value_t, size_, SOA>::operator()(const std::size_t level,
    //                                                       const interval_t &interval, const T... index)
    // {
    //     auto interval_tmp = get_interval("READ OR WRITE", level, interval, index...);
    //     return xt::view(m_data,
    //                     xt::range(interval_tmp.index + interval.start,
    //                                 interval_tmp.index + interval.end,
    //                                 interval.step));
    // }

    // template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    // template<class... T>
    // inline auto Field<mesh_t, value_t, size_, SOA>::operator()(const std::size_t level,
    //                                                       const interval_t &interval, const T... index) const
    // {
    //     auto interval_tmp = get_interval("READ", level, interval, index...);
    //     return xt::view(m_data,
    //                     xt::range(interval_tmp.index + interval.start,
    //                                 interval_tmp.index + interval.end,
    //                                 interval.step));
    // }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::array() const -> const data_type&
    {
        return m_data;
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline auto Field<mesh_t, value_t, size_, SOA>::array() -> data_type&
    {
        return m_data;
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline std::string Field<mesh_t, value_t, size_, SOA>::name() const
    {
        return m_name;
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    inline void Field<mesh_t, value_t, size_, SOA>::to_stream(std::ostream& os) const
    {
        os << "Field " << m_name << "\n";
        for_each_cell(this->mesh(), [&](auto &cell)
        {
                os << "\tlevel: " << cell.level << " coords: " << cell.center()
                    << " value: " << this->operator[](cell) << "\n";
        });
    }

    template<class mesh_t, class T, std::size_t N, bool SOA>
    inline std::ostream& operator<<(std::ostream& out, const Field<mesh_t, T, N, SOA>& field)
    {
        field.to_stream(out);
        return out;
    }


    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    const std::vector<typename Field<mesh_t, value_t, size_, SOA>::boundary_condition_t>&
    Field<mesh_t, value_t, size_, SOA>::boundary_conditions() const
    {
        return m_boundary_conditions;
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    typename Field<mesh_t, value_t, size_, SOA>::boundary_condition_t&
    Field<mesh_t, value_t, size_, SOA>::set_dirichlet(boundary_cond_t dirichlet_value)
    {
        m_boundary_conditions.emplace_back(boundary_condition_t::BCType::Dirichlet, dirichlet_value);
        return m_boundary_conditions.back();
    }

    template<class mesh_t, class value_t, std::size_t size_, bool SOA>
    auto
    Field<mesh_t, value_t, size_, SOA>::set_neumann(boundary_cond_t neumann_value) -> boundary_condition_t&
    {
        m_boundary_conditions.emplace_back(boundary_condition_t::BCType::Neumann, neumann_value);
        return m_boundary_conditions.back();
    }

    template <class value_t, std::size_t size, bool SOA=false, class mesh_t>
    auto make_field(std::string name, mesh_t& mesh)
    {
        using field_t = Field<mesh_t, value_t, size, SOA>;
        return field_t(name, mesh);
    }

    /**
     * @brief Creates a field.
     * @param name Name of the returned Field.
     * @param f Continuous function.
     * @param polynomial_degree Polynomial degree of the function (-1 if not polynomial).
    */
    template<class value_t, std::size_t size, bool SOA=false, class mesh_t, class Func>
    auto make_field(std::string name, mesh_t& mesh, Func&& f, int polynomial_degree=-1)
    {
        auto field = make_field<value_t, size, SOA, mesh_t>(name, mesh);
        field.fill(0);
        GaussLegendre gl(polynomial_degree);

        for_each_cell(mesh, [&](const auto& cell)
        {
            const double& h = cell.length;
            field[cell] = gl.quadrature(cell, f) / pow(h, mesh_t::dim);
        });
        return field;
    }

} // namespace samurai