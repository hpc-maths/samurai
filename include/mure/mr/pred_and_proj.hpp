#pragma once

#include "../field.hpp"

namespace mure
{
    template<class MRConfig>
    void mr_projection(Field<MRConfig> &field)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        auto mesh = field.mesh();
        for (std::size_t level = max_refinement_level; level >= 1; --level)
        {
            auto expr = intersection(mesh[MeshType::all_cells][level],
                                     mesh[MeshType::proj_cells][level - 1])
                            .on(level - 1);
            expr.apply_op(level - 1, projection(field));
        }
    }

    template<class MRConfig>
    void mr_prediction(Field<MRConfig> &field)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        auto mesh = field.mesh();
        for (std::size_t level = 1; level <= max_refinement_level; ++level)
        {

            if (!mesh[MeshType::cells][level].empty())
            {
                auto expr =
                    intersection(
                        difference(mesh[MeshType::all_cells][level],
                                   union_(mesh[MeshType::cells][level],
                                          mesh[MeshType::proj_cells][level])),
                        mesh.initial_mesh())
                        .on(level);
                expr.apply_op(level, prediction(field));
                if (level == 2)
                {
                    expr([&](auto &index_yz, auto &interval, auto &) {
                        std::cout << interval[0] << " " << index_yz[0] << "\n";
                        std::cout << field(level, interval[0], index_yz[0])
                                  << "\n";
                    });
                }
            }
        }
    }
}