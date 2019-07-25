#pragma once

#include <array>
#include <memory>

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "field_expression.hpp"
#include "mesh.hpp"
#include "mesh_type.hpp"

namespace mure
{
    template<class MRConfig, class value_t>
    class Field : public field_expression<Field<MRConfig, value_t>> {
      public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level =
            MRConfig::max_refinement_level;

        using value_type = value_t;
        using data_type = xt::xtensor<value_type, 1>;
        using view_type =
            decltype(xt::view(std::declval<data_type &>(), xt::range(0, 1, 1)));

        using coord_index_t = typename MRConfig::coord_index_t;
        using index_t = typename MRConfig::index_t;
        using interval_t = typename MRConfig::interval_t;

        // Field(Field const &) = default;
        // Field &operator=(Field const &) = default;

        // Field(Field &&) = default;
        // Field &operator=(Field &&) = default;

        Field(std::string name, Mesh<MRConfig> &mesh)
            : name_(name), mesh(&mesh),
              m_data(std::array<std::size_t, 1>{mesh.nb_total_cells()})
        {
            m_data.fill(0);
        }

        template<class E>
        Field &operator=(const field_expression<E> &e)
        {
            // mesh->for_each_cell(
            //     [&](auto &cell) { (*this)[cell] = e.derived_cast()(cell); });

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto subset = intersection((*mesh)[MeshType::cells][level],
                                           (*mesh)[MeshType::cells][level]);

                subset.apply_op(level, apply_expr(*this, e));
            }
            return *this;
        }

        value_type const operator()(const Cell<coord_index_t, dim> &cell) const
        {
            return m_data[cell.index];
        }

        value_type &operator()(const Cell<coord_index_t, dim> &cell)
        {
            return m_data[cell.index];
        }

        value_type const operator[](const Cell<coord_index_t, dim> &cell) const
        {
            return m_data[cell.index];
        }

        value_type &operator[](const Cell<coord_index_t, dim> &cell)
        {
            return m_data[cell.index];
        }

        template<class... T>
        auto operator()(interval_t interval, T... index) const
        {
            return xt::view(m_data, xt::range(interval.start, interval.end));
        }

        template<class... T>
        auto operator()(interval_t interval, T... index)
        {
            return xt::view(m_data, xt::range(interval.start, interval.end));
        }

        template<class... T>
        auto operator()(const std::size_t level, const interval_t &interval,
                        const T... index)
        {
            auto interval_tmp = mesh->get_interval(level, interval, index...);
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        template<class... T>
        auto operator()(const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = mesh->get_interval(level, interval, index...);
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        auto data(MeshType mesh_type) const
        {
            std::array<std::size_t, 1> shape = {mesh->nb_cells(mesh_type)};
            xt::xtensor<double, 1> output(shape);
            std::size_t index = 0;
            mesh->for_each_cell(
                [&](auto cell) { output[index++] = m_data[cell.index]; },
                mesh_type);
            return output;
        }

        auto data_on_level(std::size_t level, MeshType mesh_type) const
        {
            std::array<std::size_t, 1> shape = {
                mesh->nb_cells(level, mesh_type)};
            xt::xtensor<double, 1> output(shape);
            std::size_t index = 0;
            mesh->for_each_cell(
                level, [&](auto cell) { output[index++] = m_data[cell.index]; },
                mesh_type);
            return output;
        }

        inline auto const array() const
        {
            return m_data;
        }

        inline auto &array()
        {
            return m_data;
        }

        inline std::size_t nb_cells(MeshType mesh_type) const
        {
            return mesh->nb_cells(mesh_type);
        }

        inline std::size_t nb_cells(std::size_t level, MeshType mesh_type) const
        {
            return mesh->nb_cells(level, mesh_type);
        }

        auto const &name() const
        {
            return name_;
        }

        void to_stream(std::ostream &os) const
        {
            os << "Field " << name_ << "\n";
            mesh->for_each_cell(
                [&](auto &cell) {
                    os << cell.level << "[" << cell.center()
                       << "]:" << m_data[cell.index] << "\n";
                },
                MeshType::all_cells);
        }

      private:
        std::string name_;
        Mesh<MRConfig> *mesh;
        data_type m_data;
    };

    template<class MRConfig, class T>
    std::ostream &operator<<(std::ostream &out, const Field<MRConfig, T> &field)
    {
        field.to_stream(out);
        return out;
    }
}