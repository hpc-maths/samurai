#pragma once
#include "stencil.hpp"

namespace samurai_new
{
    template <typename TIndex, class Mesh, typename TJ, class Func>
    inline void for_each_cell_index(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, Func &&f)
    {
        auto i_start = static_cast<TIndex>(mesh.get_index(level, i.start, j));
        for(TIndex ii=0; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            f(i_start + ii);
        }
    }

    /*template <typename TIndex, class Mesh, typename TJ, typename TDirection, class Func>
    inline void for_each_index_and_nghb_index(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, const TDirection& nghb_direction, Func &&f)
    {
        auto i_start = static_cast<TIndex>(mesh.get_index(level, i.start, j));
        auto i_nghb_start = i_start;
        if (nghb_direction[1] == 0) // the neighbour is on the same row
        {
            i_nghb_start += nghb_direction[0];
        }
        else
        {
            i_nghb_start = static_cast<TIndex>(mesh.get_index(level, i.start + nghb_direction[0], j + nghb_direction[1]));
        }
        
        for(TIndex ii=0; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            f(i_start + ii, i_nghb_start + ii);
        }
    }*/

    /*template <typename TIndex, class Mesh, typename TJ, class TStencil, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, const TStencil& stencil, Func &&f)
    {
        TIndex i_stencil[stencil.shape()[0]+1];
        i_stencil[0] = static_cast<TIndex>(mesh.get_index(level, i.start, j)); // center of the stencil
        for (std::size_t id = 0; id<stencil.shape()[0]; ++id)
        {
            auto d = xt::view(stencil, id);
            if (d[1] == 0) // same row as the stencil center
            {
                i_stencil[id+1] = i_stencil[0] + d[0];
            }
            else
            {
                i_stencil[id+1] = static_cast<TIndex>(mesh.get_index(level, i.start + d[0], j + d[1]));
            }
        }

        f(i_stencil);
        for(TIndex ii=1; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            for (int cell = 0; cell < stencil.shape()[0]+1; ++cell)
            {
                i_stencil[cell]++;
            }
            f(i_stencil);
        }
    }*/

    template <typename DesiredIndexType, class Mesh, typename TIndex, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t i, const TIndex& index, StencilType& stencil, Func &&f)
    {
        stencil.init(mesh, level, i, index);
        f(stencil.indices());
        for(DesiredIndexType ii=1; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
        {
            stencil.move_next();
            f(stencil.indices());
        }
    }

    template <typename DesiredIndexType, class Mesh, class Set, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, Set& set, std::size_t level, StencilType& stencil, Func &&f)
    {
        samurai::for_each_interval(set[level], [&](std::size_t level, const auto& i, const auto& index)
        {
            /*stencil.init(mesh, level, i, index);
            f(stencil.indices());
            for(DesiredIndexType ii=1; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
            {
                stencil.move_next();
                f(stencil.indices());
            }*/
            for_each_stencil<DesiredIndexType>(mesh, level, i, index, stencil, f);
        });
    }

    template <typename DesiredIndexType, class Mesh, class Set, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, Set& set, StencilType& stencil, Func &&f)
    {
        for_each_level(set, [&](std::size_t level)
        {
            for_each_stencil<DesiredIndexType>(mesh, set, level, stencil, f);
        });
    }

    template <typename DesiredIndexType, class Mesh, class Set, class Func>
    inline void for_each_cell_index(const Mesh& mesh, Set& set, Func &&f)
    {
        samurai::for_each_interval(set, [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0];
            auto i_j_start = static_cast<DesiredIndexType>(mesh.get_index(level, i.start, j));
            for(DesiredIndexType ii=0; static_cast<DesiredIndexType>(ii<i.size()); ++ii)
            {
                f(i_j_start + ii);
            }
        });
    }
}