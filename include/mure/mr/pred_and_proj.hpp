#pragma once

#include <spdlog/spdlog.h>

#include "../field.hpp"

namespace mure
{
    template<class MRConfig>
    inline void mr_projection(Field<MRConfig> &field)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        spdlog::info("Make projection");
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
    inline void mr_prediction(Field<MRConfig> &field)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        spdlog::info("Make prediction");

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
            }
        }
    }


    template<class MRConfig>
    inline void mr_prediction_for_debug(Field<MRConfig> &field, std::size_t mx_lev)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        std::cout<<"\n\nThe level is = "<<mx_lev<<std::endl;

        spdlog::info("Make prediction");

        auto mesh = field.mesh();

        auto set = intersection(mesh[MeshType::cells][mx_lev],
                                mesh[MeshType::cells][mx_lev]);

        set.apply_op(mx_lev, prediction(field));
    }
}
