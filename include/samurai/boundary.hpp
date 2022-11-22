#pragma once
#include <utility>
#include "stencil.hpp"

namespace samurai
{
    /*template<class Vector>
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
    }*/

    template <class Mesh, class Vector>
    auto in_boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells][level];
        auto& domain = mesh.domain();

        auto max_level = mesh[mesh_id_t::cells].max_level();
        auto one_interval = 1<<(max_level-level);

        return difference(cells, translate(domain, -one_interval * direction)).on(level);
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_interval_on_boundary(const Mesh& mesh, std::size_t level, const Stencil<stencil_size, Mesh::dim>& stencil, const std::array<double, stencil_size>& coefficients, Func &&func)
    {
        typename Mesh::mesh_interval_t mesh_interval(level);
        for (unsigned int is = 0; is<stencil_size; ++is)
        {
            auto direction = xt::view(stencil, is);
            if (xt::any(direction)) // if (direction != 0)
            {
                double coeff = coefficients[is];
                auto boundary = in_boundary(mesh, level, direction);
                boundary([&](auto& i, auto& index)
                {
                    mesh_interval.i = i;
                    mesh_interval.index = index;
                    func(mesh_interval, direction, coeff);
                });
            }
        }
    }

    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_interval_on_boundary(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil, GetCoeffsFunc&& get_coefficients, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells];
        for(std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!cells[level].empty())
            {
                double h = 1./(1<<level);
                auto coeffs = get_coefficients(h);
                for_each_interval_on_boundary(mesh, level, stencil, coeffs, std::forward<Func>(func));
            }
        }
    }

    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_cell_on_boundary(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil, GetCoeffsFunc&& get_coefficients, Func &&func)
    {
        for_each_interval_on_boundary(mesh, stencil, get_coefficients,
        [&] (auto& mesh_interval, auto& stencil_vector, double out_coeff)
        {
            for_each_cell(mesh, mesh_interval.level, mesh_interval.i, mesh_interval.index, [&](auto& cell)
            {
                func(cell, stencil_vector, out_coeff);
            });
        });
    }


    /*
    template <class Mesh, class Func>
    void out_boundary(const Mesh& mesh, std::size_t level, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        static constexpr std::size_t n_cart_dir = (2*dim); // number of Cartesian directions

        MeshInterval<Mesh> mesh_interval(level);

        const Stencil<n_cart_dir, dim> cart_directions = cartesian_directions<dim>();

        for (std::size_t id1 = 0; id1<n_cart_dir; ++id1)
        {
            // Cartesian direction
            auto d1 = xt::view(cart_directions, id1);

            // Boundary in the direction d1
            auto boundary_d1 = difference(translate(mesh[mesh_id_t::cells][level], d1), mesh.domain());

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
                auto boundary_d2 = difference(translate(mesh[mesh_id_t::cells][level], d2), mesh.domain());

                auto out_vect_d1d2 = out_vect_d1 + d2;

                // Corners between boundary_d1 and boundary_d2
                auto boundaries_d1d2 = difference(translate(mesh[mesh_id_t::cells][level], out_vect_d1d2), mesh.domain());
                auto corners_d1d2 = difference(difference(boundaries_d1d2, boundary_d1), boundary_d2);

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
                    auto boundary_d3 = difference(translate(mesh[mesh_id_t::cells][level], d3), mesh.domain());

                    auto out_vect_d1d2d3 = out_vect_d1d2 + d3;

                    // Corners between boundary_d1, boundary_d2 and boundary_d3
                    auto boundaries_d1d2d3 = difference(translate(mesh[mesh_id_t::cells][level], out_vect_d1d2d3), mesh.domain());
                    auto corners_d1d2d3 = difference(difference(difference(boundaries_d1d2d3, boundary_d1), boundary_d2), boundary_d3);

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
    }*/

    /*template <class Mesh, class Func>
    void interior(const Mesh& mesh, std::size_t level, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        static constexpr std::size_t n_cart_dir = (2*dim); // number of Cartesian directions

        const auto& cells = mesh[mesh_id_t::cells][level];
        auto& domain = mesh.domain();

        const Stencil<dim, n_cart_dir> cart_directions = cartesian_directions<dim>();
        MeshInterval<Mesh> mesh_interval(level);


        //auto bdry0 = translate(mesh.domain(), xt::view(cart_directions, 0));
        //auto minus_bdry0 = intersection(mesh[mesh_id_t::cells][level], bdry0);


        unsigned int d = 0;
        if constexpr(dim >= 1)
        {
            auto minus_bdry0 = intersection(      cells, translate(domain, xt::view(cart_directions, d++)));
            auto minus_bdry1 = intersection(minus_bdry0, translate(domain, xt::view(cart_directions, d++)));
            if constexpr(dim == 1)
            {
                minus_bdry1([&](const auto& i, const auto& index)
                {
                    mesh_interval.i = i;
                    mesh_interval.index = index;
                    func(mesh_interval);
                });
            }
            if constexpr(dim >= 2)
            {
                auto minus_bdry2 = intersection(minus_bdry1, translate(domain, xt::view(cart_directions, d++)));
                auto minus_bdry3 = intersection(minus_bdry2, translate(domain, xt::view(cart_directions, d++)));
                if constexpr(dim == 2)
                {
                    minus_bdry3([&](const auto& i, const auto& index)
                    {
                        mesh_interval.i = i;
                        mesh_interval.index = index;
                        func(mesh_interval);
                    });
                }
                if constexpr(dim >= 3)
                {
                    auto minus_bdry4 = intersection(minus_bdry3, translate(domain, xt::view(cart_directions, d++)));
                    auto minus_bdry5 = intersection(minus_bdry4, translate(domain, xt::view(cart_directions, d++)));
                    if constexpr(dim == 3)
                    {
                        minus_bdry5([&](const auto& i, const auto& index)
                        {
                            mesh_interval.i = i;
                            mesh_interval.index = index;
                            func(mesh_interval);
                        });
                    }
                    if constexpr(dim > 3)
                    {
                        static_assert(dim < 4, "interior() not implemented for dim > 3.");
                    }
                }
            }
        }
    }

    template <class Mesh, class Func>
    inline void interior(const Mesh& mesh, Func &&func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells];
        for(std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!cells[level].empty())
            {
                interior(mesh, level, std::forward<Func>(func));
            }
        }
    }*/
}