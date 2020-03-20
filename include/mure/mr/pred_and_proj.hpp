#pragma once

#include <spdlog/spdlog.h>

#include "../field.hpp"

namespace mure
{
    template<class Field>
    inline void mr_projection(Field &field)
    {
        spdlog::info("Make projection");

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= min_level; --level)
        {
            auto expr = intersection(mesh[MeshType::all_cells][level],
                                     mesh[MeshType::proj_cells][level - 1])
                       .on(level - 1);

            expr.apply_op(level - 1, projection(field));
        }
    }

    template<class Field>
    inline void mr_prediction(Field &field)
    {
        spdlog::info("Make prediction");

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            if (!mesh[MeshType::cells][level].empty())
            {
                auto expr = intersection(difference(mesh[MeshType::all_cells][level],
                                         union_(mesh[MeshType::cells][level],
                                                mesh[MeshType::proj_cells][level])),
                                         mesh.initial_mesh())
                           .on(level);

                expr.apply_op(level, prediction(field));
            }
        }
    }


    template<class Field>
    inline void mr_prediction_for_debug(Field &field, std::size_t mx_lev)
    {
        std::cout<<"\n\nThe level is = "<<mx_lev<<std::endl;

        spdlog::info("Make prediction");

        auto mesh = field.mesh();

        auto set = intersection(mesh[MeshType::cells][mx_lev],
                                mesh[MeshType::cells][mx_lev]);

        set.apply_op(mx_lev, prediction(field));
    }
}
