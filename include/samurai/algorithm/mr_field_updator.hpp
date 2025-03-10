// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../cell_flag.hpp"

namespace samurai
{

    template <class Tag, class Field>
    class MrFieldUpdator
    {
      public:

        MrFieldUpdator()
        {
        }

        template <class... Fields>
        bool update(const Tag& tag, Field& field, Fields&... other_fields);

      private:

        using raw_field_t                = std::decay_t<Field>;
        using mesh_t                     = typename raw_field_t::mesh_t;
        static constexpr std::size_t dim = mesh_t::dim;
        using mesh_id_t                  = typename raw_field_t::mesh_t::mesh_id_t;
        using size_type                  = typename raw_field_t::size_type;
        using interval_t                 = typename mesh_t::interval_t;
        using value_t                    = typename interval_t::value_t;
        using lca_type                   = typename raw_field_t::mesh_t::lca_type;
        using ca_type                    = typename raw_field_t::mesh_t::ca_type;
        using coord_type                 = typename lca_type::coord_type;

        ca_type m_ca_add_m;
        ca_type m_ca_add_p;
        ca_type m_ca_remove_m;
        ca_type m_ca_remove_p;
        ca_type m_new_ca;

        std::vector<value_t> m_add_p_x;
        std::vector<coord_type> m_add_p_yz;

        std::vector<size_t> m_add_p_idx;
    };

    template <class Tag, class Field>
    template <class... Fields>
    bool MrFieldUpdator<Tag, Field>::update(const Tag& tag, Field& field, Fields&... other_fields)
    {
        using unsigned_value_t = std::make_unsigned_t<value_t>;

        const auto& mesh = tag.mesh();

        if constexpr (dim > 1)
        {
            m_add_p_x.clear();
            if constexpr (dim > 2)
            {
                m_add_p_yz.clear();
            }
        }

        for (std::size_t level = 0; level <= ca_type::max_size; ++level)
        {
            m_ca_add_m[level].clear();
            m_ca_add_p[level].clear();
            m_ca_remove_m[level].clear();
            m_ca_remove_p[level].clear();
            m_new_ca[level].clear();
        }

        for (std::size_t level = mesh[mesh_id_t::cells].min_level(); level <= mesh[mesh_id_t::cells].max_level(); ++level)
        {
            const auto begin = mesh[mesh_id_t::cells][level].cbegin();
            const auto end   = mesh[mesh_id_t::cells][level].cend();
            for (auto it = begin; it != end; ++it)
            {
                const auto& x_interval = *it;
                const auto& yz         = it.index();

                const bool is_yz_even = dim == 1 or xt::all(not xt::eval(yz % 2));

                for (value_t x = x_interval.start; x < x_interval.end; ++x)
                {
                    const size_type itag         = static_cast<size_type>(x_interval.index) + static_cast<unsigned_value_t>(x);
                    const bool refine            = tag[itag] & static_cast<int>(CellFlag::refine);
                    const bool coarsenAndNotKeep = tag[itag] & static_cast<int>(CellFlag::coarsen)
                                               and not(tag[itag] & static_cast<int>(CellFlag::keep));

                    if (refine and level < mesh.max_level())
                    {
                        m_ca_remove_p[level].add_point_back(x, yz);
                        if constexpr (dim == 1)
                        {
                            m_ca_add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, {});
                        }
                        else if constexpr (dim == 2)
                        {
                            m_add_p_x.push_back(x);
                        }
                        else
                        {
                            static_nested_loop<dim - 1, 0, 2>(
                                [this, &x, &yz](const coord_type& stencil)
                                {
                                    m_add_p_x.push_back(2 * x);
                                    m_add_p_yz.emplace_back(2 * yz + stencil);
                                });
                        }
                    }
                    else if ((coarsenAndNotKeep and level > mesh.min_level())
                    {
                        if (x % 2 == 0 and is_yz_even) // should be modified when using load balencing.
                        {
                            m_ca_add_m[level - 1].add_point_back(x >> 1, yz >> 1); // add cell / 2 at level-1
                        }
                        m_ca_remove_m[level].add_point_back(x, yz);
                    }

                } // end for x
                if constexpr (dim == 2)
                {
                    if ((it + 1 == end) or (it + 1).index()[dim - 2] != yz[dim - 2])
                    {
                        coord_type stencil;
                        for (stencil[dim - 2] = 0; stencil[dim - 2] != 2; ++stencil[dim - 2])
                        {
                            for (const auto& x : m_add_p_x)
                            {
                                m_ca_add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, 2 * yz + stencil);
                            }
                        }
                        m_add_p_x.clear();
                    }
                }
                else if constexpr (dim > 2)
                {
                    if ((it + 1 == end) or (it + 1).index()[dim - 2] != yz[dim - 2])
                    {
                        sort_indexes(
                            m_add_p_yz,
                            [](const coord_type& lhs, const coord_type& rhs) -> bool
                            {
                                for (size_t i = dim - 2; i != 0; --i)
                                {
                                    if (lhs[i] < rhs[i])
                                    {
                                        return true;
                                    }
                                    else if (lhs[i] > rhs[i])
                                    {
                                        return false;
                                    }
                                }
                                return lhs[0] < rhs[0];
                            },
                            m_add_p_idx);
                        for (const auto i : m_add_p_idx)
                        {
                            m_ca_add_p[level + 1].add_interval_back({m_add_p_x[i], m_add_p_x[i] + 2}, m_add_p_yz[i]);
                        }
                        m_add_p_x.clear();
                        m_add_p_yz.clear();
                    }
                }
            }
        }

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            auto set = difference(union_(mesh[mesh_id_t::cells][level], m_ca_add_m[level], m_ca_add_p[level]),
                                  union_(m_ca_remove_m[level], m_ca_remove_p[level]));
            set(
                [this, &level](const auto& x_interval, const auto& yz)
                {
                    m_new_ca[level].add_interval_back({x_interval.start, x_interval.end}, yz);
                });
        }
        mesh_t new_mesh{m_new_ca, mesh};

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif
        {
            return true;
        }
        detail::update_fields(new_mesh, field, other_fields...);

        field.mesh().swap(new_mesh);

        return false;
    }

}
