#pragma once
#include <samurai/algorithm.hpp>
#include "stencil.hpp"

namespace samurai_new
{
    template <std::size_t dim>
    inline StencilShape<dim, 2*dim> cartesian_directions()
    {
        // The order is important: the opposite of a vector must be located 'dim' indices after.
        assert(false && "Not implemented in N-D");
        return StencilShape<dim, 2*dim>();
    }
    template<> 
    inline StencilShape<1, 2> cartesian_directions<1>()
    {
        //                       left, right
        return StencilShape<1, 2>{{-1}, {1}};
    }
    template<> 
    inline StencilShape<2, 4> cartesian_directions<2>()
    {
        //                        bottom,   right,  top,    left      (the order is important)
        return StencilShape<2, 4>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    }
    template<> 
    inline StencilShape<3, 6> cartesian_directions<3>()
    {
        //                         bottom,   front,   right,    top,     back,     left     (the order is important)
        return StencilShape<3, 6>{{0,0,-1}, {0,1,0}, {1,0,0}, {0,0,1}, {0,-1,0}, {-1,0,0}};
    }

    /*template<class Vector>
    inline bool is_cartesian_direction(const Vector& v)
    {
        bool only_one_non_zero = false;
        for (std::size_t i=0; i<v.shape()[0]; ++i)
        {
            if (v[i] != 0)
            {
                if (!only_one_non_zero)
                {
                    only_one_non_zero = true;
                }
                else
                {
                    only_one_non_zero = false; // second non-zero found
                    break;
                }
            }
            
        }
        return only_one_non_zero;
    }*/

    template<class Vector>
    inline unsigned int number_of_zeros(const Vector& v)
    {
        unsigned int n_zeros = 0;
        for (std::size_t i=0; i<v.shape()[0]; ++i)
        {
            n_zeros += v[i] == 0 ? 1 : 0;
        }
        return n_zeros;
    }


    template <class Mesh, class Func>
    void out_boundary(const Mesh& mesh, std::size_t level, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        static constexpr std::size_t n_cart_dir = (2*dim); // number of Cartesian directions

        const StencilShape<dim, n_cart_dir> cart_directions = cartesian_directions<dim>();

        for (std::size_t id1 = 0; id1<n_cart_dir; ++id1)
        {
            // Cartesian direction
            auto d1 = xt::view(cart_directions, id1);
            // Next Cartesian direction
            std::size_t id2 = id1+1;
            if (id2 == n_cart_dir)
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