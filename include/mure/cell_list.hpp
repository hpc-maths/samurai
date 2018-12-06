#pragma once

#include <xtensor/xfixed.hpp>
#include "level_cell_list.hpp"

namespace mure
{
    template<typename MRConfig>
    class CellList
    {
        static constexpr int dim = MRConfig::dim;
        enum {max_refinement_level = MRConfig::max_refinement_level};

    public:
        CellList() = default;

        LevelCellList<MRConfig> const& operator[](int i) const
        {
            return m_cells[i];
        }

        LevelCellList<MRConfig>& operator[](int i)
        {
            return m_cells[i];
        }

    private:
        xt::xtensor_fixed<LevelCellList<MRConfig>, xt::xshape<max_refinement_level + 1>> m_cells;
    };
}