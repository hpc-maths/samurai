#pragma once

#include <xtensor/xfixed.hpp>

#include "cell_list.hpp"
#include "level_cell_array.hpp"

namespace mure
{
    template<typename MRConfig>
    class CellArray {
      public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        inline CellArray(const CellList<MRConfig> &dcl = {})
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                m_cells[level] = dcl[level];
            }
        }

        inline LevelCellArray<dim> const &operator[](std::size_t i) const
        {
            return m_cells[i];
        }

        inline LevelCellArray<dim> &operator[](std::size_t i)
        {
            return m_cells[i];
        }

        inline std::size_t nb_cells() const
        {
            std::size_t size = 0;
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                size += m_cells[level].nb_cells();
            }
            return size;
        }

        inline std::size_t max_level() const
        {
            for (int level = max_refinement_level; level >= 0; --level)
            {
                if (!m_cells[level].empty())
                    return level;
            }
            return 0;
        }

        inline std::size_t min_level() const
        {
            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (!m_cells[level].empty())
                    return level;
            }
            return max_refinement_level + 1;
        }

        template<class Func>
        inline void for_each_cell(Func &&func) const
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                if (!m_cells[level].empty())
                {
                    m_cells[level].for_each_cell(level,
                                                 std::forward<Func>(func));
                }
            }
        }

        template<class Func>
        inline void for_each_cell(std::size_t level, Func &&func) const
        {
            assert(level <= max_refinement_level and level >= 0);
            if (!m_cells[level].empty())
            {
                m_cells[level].for_each_cell(level, std::forward<Func>(func));
            }
        }

        inline void to_stream(std::ostream &os) const
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                if (!m_cells[level].empty())
                {
                    os << "level " << level << "\n";
                    m_cells[level].to_stream(os);
                    os << "\n";
                }
            }
        }

      private:
        xt::xtensor_fixed<LevelCellArray<dim>,
                          xt::xshape<max_refinement_level + 1>>
            m_cells;
    };

    template<class MRConfig>
    inline std::ostream &operator<<(std::ostream &out,
                             const CellArray<MRConfig> &cell_array)
    {
        cell_array.to_stream(out);
        return out;
    }
}