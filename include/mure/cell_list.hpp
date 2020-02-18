#pragma once

#include <xtensor/xfixed.hpp>

#include "level_cell_list.hpp"

namespace mure
{
    template<class MRConfig>
    class CellList {
      public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level =
            MRConfig::max_refinement_level;

        inline CellList()
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                m_cells[level].set_level(level);
            }
        }

        inline LevelCellList<dim> const &operator[](std::size_t i) const
        {
            return m_cells[i];
        }

        inline LevelCellList<dim> &operator[](std::size_t i)
        {
            return m_cells[i];
        }

        inline void to_stream(std::ostream &os) const
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                os << "level " << level << "\n";
                m_cells[level].to_stream(os);
                os << "\n";
            }
        }

      private:
        xt::xtensor_fixed<LevelCellList<dim>,
                          xt::xshape<max_refinement_level + 1>>
            m_cells;
    };

    template<class MRConfig>
    inline std::ostream &operator<<(std::ostream &out,
                             const CellList<MRConfig> &cell_list)
    {
        cell_list.to_stream(out);
        return out;
    }
}