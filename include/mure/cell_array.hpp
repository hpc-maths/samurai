#pragma once

#include <array>

#include "level_cell_array.hpp"

namespace mure
{
    template<typename MRConfig>
    class CellArray
    {
        enum {dim = MRConfig::dim};
        enum {max_refinement_level = MRConfig::max_refinement_level};

    public:
        CellArray(){}

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

        std::array<LevelCellArray<MRConfig>, max_refinement_level + 1> m_cells;
    };
}