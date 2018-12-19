#pragma once

#include <array>

#include <xtensor/xfixed.hpp>

#include "cell.hpp"
#include "mesh.hpp"

namespace mure
{
    template<class MRConfig>
    class Field
    {
    public:
        static constexpr auto dim = MRConfig::dim;
        using coord_index_t = typename MRConfig::coord_index_t;
        using index_t = typename MRConfig::index_t;

        Field(std::string name, Mesh<MRConfig>& mesh)
            : name_(name), mesh(mesh),
              m_data(std::array<std::size_t, 1>{mesh.nb_total_cells()})
        {}

        double const operator[](Cell<coord_index_t, index_t, dim> cell) const
        {
            return m_data[cell.index];
        }

        double& operator[](Cell<coord_index_t, index_t, dim> cell)
        {
            return m_data[cell.index];
        }

        auto data() const
        {
            std::array<std::size_t, 1> shape = {mesh.nb_cells()};
            xt::xtensor<double, 1> output(shape);
            std::size_t index = 0;
            mesh.for_each_cell([&](auto cell)
            {
                output[index++] = m_data[cell.index];
            });
            return output;
        }

        inline std::size_t nb_cells() const
        {
            return mesh.nb_cells();
        }

        auto const& name() const
        {
            return name_;
        }

    private:
        std::string name_;
        xt::xtensor<double, 1> m_data;
        xt::xtensor<double, 1> m_work;
        Mesh<MRConfig>& mesh;
    };
}