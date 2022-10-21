#pragma once
#include <cstddef>

namespace samurai_new
{
    template <typename TIndex, class Mesh, typename TJ, class Func>
    inline void for_each_index(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, Func &&f)
    {
        auto i_start = static_cast<TIndex>(mesh.get_index(level, i.start, j));
        for(TIndex ii=0; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            auto index = i_start + ii;
            f(index);
        }
    }

    template <typename TIndex, class Mesh, typename TJ, typename TDirection, class Func>
    inline void for_each_index_and_nghb_index(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, const TDirection& nghb_direction, Func &&f)
    {
        auto i_start      = static_cast<TIndex>(mesh.get_index(level, i.start                    , j                    ));
        auto i_nghb_start = static_cast<TIndex>(mesh.get_index(level, i.start + nghb_direction[0], j + nghb_direction[1]));
        for(TIndex ii=0; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            auto index      = i_start      + ii;
            auto index_nghb = i_nghb_start + ii;
            f(index, index_nghb);
        }
    }

    template <typename TIndex, class Mesh, typename TJ, class TStencil, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const TJ j, const TStencil& stencil, Func &&f)
    {
        TIndex i_stencil[stencil.shape()[0]+1];
        i_stencil[0] = static_cast<TIndex>(mesh.get_index(level, i.start, j)); // center of the stencil
        for (std::size_t id = 0; id<stencil.shape()[0]; ++id)
        {
            auto d = xt::view(stencil, id);
            if (d[1] == 0) // same row as the stencil center
                i_stencil[id+1] = i_stencil[0] + d[0];
            else
                i_stencil[id+1] = static_cast<TIndex>(mesh.get_index(level, i.start + d[0], j + d[1]));
        }

        f(i_stencil);
        for(TIndex ii=1; ii<static_cast<TIndex>(i.size()); ++ii)
        {
            for (int cell = 0; cell < stencil.shape()[0]+1; ++cell)
                i_stencil[cell]++;
            f(i_stencil);
        }
    }
}