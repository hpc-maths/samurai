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

        void to_stream(std::ostream &os) const
        {
            for(std::size_t level=0; level <= max_refinement_level; ++level)
            {
                os << "level " << level << "\n";
                m_cells[level].to_stream(os);
                os << "\n";
            }
        }

    private:

        xt::xtensor_fixed<LevelCellArray<MRConfig>, xt::xshape<max_refinement_level + 1>> m_cells;
    };

    template<class MRConfig>
    std::ostream& operator<<(std::ostream& out, const CellArray<MRConfig>& cell_array)
    {
        cell_array.to_stream(out);
        return out;
    }

}