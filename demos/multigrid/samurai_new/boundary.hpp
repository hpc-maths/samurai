#pragma once
#include <samurai/algorithm.hpp>

namespace samurai_new
{


    template <class Mesh, class Func>
    auto boundary(const Mesh& mesh, std::size_t level, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        // Cartesian directions: bottom, right, top, left (the order is important)
        xt::xtensor_fixed<int, xt::xshape<4, 2>> cart_directions{{0, -1}, {1, 0}, {0, 1}, {-1, 0}};

        for (std::size_t id1 = 0; id1<cart_directions.shape()[0]; ++id1)
        {
            // Cartesian direction
            auto d1 = xt::view(cart_directions, id1);
            // Next Cartesian direction
            std::size_t id2 = id1+1;
            if (id2 == cart_directions.shape()[0])
                id2 = 0;
            auto d2 = xt::view(cart_directions, id2);

            // Boundaries in the direction d1 and d2
            auto boundary_d1 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], d1), mesh.domain());
            auto boundary_d2 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], d2), mesh.domain());

            // Corners between boundary_d1 and boundary_d2
            auto diag = d1 + d2;
            auto boundaries_d1d2 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], diag), mesh.domain());
            auto corners = samurai::difference(samurai::difference(boundaries_d1d2, boundary_d1), boundary_d2);

            // Apply func to boundary_d1
            boundary_d1([&](const auto& i, const auto& index)
            {
                func(i, index, d1);
            });

            // Apply func to corners
            corners([&](const auto& i, const auto& index)
            {
                func(i, index, diag);
            });
        }

        
    }
}