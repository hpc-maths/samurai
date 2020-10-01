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
            // if (!mesh[MeshType::cells][level].empty())
            // {
            //     auto expr = intersection(difference(mesh[MeshType::all_cells][level],
            //                              union_(mesh[MeshType::cells][level],
            //                                     mesh[MeshType::proj_cells][level])),
            //                              mesh.initial_mesh())
            //                .on(level);

            //     expr.apply_op(level, prediction(field));
            // }

            // if (!mesh[MeshType::all_cells][level].empty() && !mesh[MeshType::overleaves][level].empty() && !mesh[MeshType::cells][level].empty())
            // {
                // We eliminate the overleaves from the computation since they 
                // are done separately
                
                auto expr = difference(intersection(difference(mesh[MeshType::all_cells][level],
                                         union_(mesh[MeshType::cells][level],
                                                mesh[MeshType::proj_cells][level])),
                                         mesh.initial_mesh()), difference(mesh[MeshType::overleaves][level], union_(mesh[MeshType::union_cells][level], mesh[MeshType::cells_and_ghosts][level])))
                           .on(level);

                expr.apply_op(level, prediction(field));

            // }
        }
    }
    

    template<class Field>
    inline void mr_prediction_overleaves(Field &field)
    {
        spdlog::info("Make prediction on the overleaves which are not already available");

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
       
            if (!mesh[MeshType::overleaves][level].empty()) {
   
                // These are the overleaves which are nothing else
                // because when this procedure is called all the rest
                // should be already with the right value.
                // auto overleaves_to_predict = 
                //     difference(difference(mesh[MeshType::overleaves][level], mesh[MeshType::cells][level]), mesh[MeshType::proj_cells][level]);
                auto overleaves_to_predict = 
                    difference(difference(mesh[MeshType::overleaves][level], mesh[MeshType::cells_and_ghosts][level]), mesh[MeshType::proj_cells][level]);

                overleaves_to_predict.apply_op(level, prediction(field));


                // overleaves_to_predict([&](auto, auto &interval, auto) {
                //     auto k = interval[0]; 
                //     std::cout<<std::endl<<"[OL Update] Level "<<level<<" Coordinate = "<<k<<std::flush;

                // });

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
