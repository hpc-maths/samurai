#pragma once

#include <xtensor/xfixed.hpp>

#include "level_cell_array.hpp"

namespace mure
{
    template<typename MRConfig>
    class CellArray
    {
    public:

        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        CellArray(const CellList<MRConfig>& dcl)
        {
            for(int level = 0; level <= (int)max_refinement_level; ++level)
            {
                m_cells[level] = dcl[level];
            }
        }

        LevelCellArray<MRConfig> const& operator[](int i) const
        {
            return m_cells[i];
        }

        LevelCellArray<MRConfig>& operator[](int i)
        {
            return m_cells[i];
        }

    private:

        xt::xtensor_fixed<LevelCellArray<MRConfig>, xt::xshape<max_refinement_level + 1>> m_cells;
    };
}