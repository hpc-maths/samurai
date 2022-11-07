#pragma once
#include <samurai/algorithm.hpp>
#include <utility>
#include "stencil.hpp"

namespace samurai_new
{
    template <std::size_t dim>
    inline StencilShape<dim, 2*dim> cartesian_directions()
    {
        static_assert((dim >= 1 && dim <=3), "cartesian_directions() not implemented in this dimension");

        // !!! The order is important: the opposite of a vector must be located 'dim' indices after.
        if constexpr (dim == 1)
        {
            //                       left, right
            return StencilShape<1, 2>{{-1}, {1}};
        }
        else if constexpr (dim == 2)
        {
            //                        bottom,   right,  top,    left
            return StencilShape<2, 4>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        }
        else if constexpr (dim == 3)
        {
            //                         bottom,   front,   right,    top,     back,     left
            return StencilShape<3, 6>{{0,0,-1}, {0,1,0}, {1,0,0}, {0,0,1}, {0,-1,0}, {-1,0,0}};
        }
        return StencilShape<dim, 2*dim>();
    }

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

    template<class Vector>
    inline bool is_cartesian_direction(const Vector& v)
    {
        std::size_t dim = v.shape()[0];
        auto n_zeros = number_of_zeros(v);
        return (dim == 0 || n_zeros == dim-1);
    }


    template <class Mesh, class Func>
    void out_boundary(const Mesh& mesh, std::size_t level, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        static constexpr std::size_t n_cart_dir = (2*dim); // number of Cartesian directions

        MeshInterval<Mesh> mesh_interval(level);

        const StencilShape<dim, n_cart_dir> cart_directions = cartesian_directions<dim>();

        for (std::size_t id1 = 0; id1<n_cart_dir; ++id1)
        {
            // Cartesian direction
            auto d1 = xt::view(cart_directions, id1);

            // Boundary in the direction d1
            auto boundary_d1 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], d1), mesh.domain());

            // Outwards normal vector
            auto out_vect_d1 = d1;

            // Apply func to boundary_d1
            boundary_d1([&](const auto& i, const auto& index)
            {
                mesh_interval.i = i;
                mesh_interval.index = index;
                func(mesh_interval, out_vect_d1);
            });

            if constexpr(dim > 1)
            {
                // Next Cartesian direction
                std::size_t id2 = id1+1;
                if (id2 == n_cart_dir)
                    id2 = 0;
                auto d2 = xt::view(cart_directions, id2);

                // Boundary in the direction d2
                auto boundary_d2 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], d2), mesh.domain());

                auto out_vect_d1d2 = out_vect_d1 + d2;

                // Corners between boundary_d1 and boundary_d2
                auto boundaries_d1d2 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], out_vect_d1d2), mesh.domain());
                auto corners_d1d2 = samurai::difference(samurai::difference(boundaries_d1d2, boundary_d1), boundary_d2);

                // Apply func to corners
                corners_d1d2([&](const auto& i, const auto& index)
                {
                    mesh_interval.i = i;
                    mesh_interval.index = index;
                    func(mesh_interval, out_vect_d1d2);
                });

                if constexpr(dim > 2)
                {
                    // Next Cartesian direction
                    std::size_t id3 = id2+1;
                    if (id3 == n_cart_dir)
                        id3 = 0;
                    auto d3 = xt::view(cart_directions, id3);

                    // Boundary in the direction d3
                    auto boundary_d3 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], d3), mesh.domain());

                    auto out_vect_d1d2d3 = out_vect_d1d2 + d3;

                    // Corners between boundary_d1, boundary_d2 and boundary_d3
                    auto boundaries_d1d2d3 = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], out_vect_d1d2d3), mesh.domain());
                    auto corners_d1d2d3 = samurai::difference(samurai::difference(samurai::difference(boundaries_d1d2d3, boundary_d1), boundary_d2), boundary_d3);

                    // Apply func to corners
                    corners_d1d2d3([&](const auto& i, const auto& index)
                    {
                        mesh_interval.i = i;
                        mesh_interval.index = index;
                        func(mesh_interval, out_vect_d1d2d3);
                    });

                    if constexpr(dim > 3)
                    {
                        static_assert(dim < 4, "out_boundary() not implemented for dim > 3.");
                    }
                }
            }
        }
    }


    template <class Mesh, class Func>
    inline void out_boundary(const Mesh& mesh, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells];
        for(std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!cells[level].empty())
            {
                out_boundary(mesh, level, std::forward<Func>(func));
            }
        }
    }
}