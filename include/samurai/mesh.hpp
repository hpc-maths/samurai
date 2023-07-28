// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <fmt/format.h>

#include "box.hpp"
#include "cell_array.hpp"
#include "cell_list.hpp"

#include "subset/subset_op.hpp"

namespace samurai
{

    template <class CellArray, class MeshID>
    struct MeshIDArray : private std::array<CellArray, static_cast<std::size_t>(MeshID::count)>
    {
        static constexpr std::size_t size = static_cast<std::size_t>(MeshID::count);
        using base_type                   = std::array<CellArray, size>;
        using base_type::operator[];

        inline const CellArray& operator[](MeshID mesh_id) const
        {
            return operator[](static_cast<std::size_t>(mesh_id));
        }

        inline CellArray& operator[](MeshID mesh_id)
        {
            return operator[](static_cast<std::size_t>(mesh_id));
        }
    };

    template <class D, class Config>
    class Mesh_base
    {
      public:

        using config = Config;

        static constexpr std::size_t dim                  = config::dim;
        static constexpr std::size_t max_refinement_level = config::max_refinement_level;

        using mesh_id_t  = typename config::mesh_id_t;
        using interval_t = typename config::interval_t;
        using value_t    = typename interval_t::value_t;
        using index_t    = typename interval_t::index_t;

        using cl_type  = CellList<dim, interval_t, max_refinement_level>;
        using lcl_type = typename cl_type::lcl_type;

        using ca_type  = CellArray<dim, interval_t, max_refinement_level>;
        using lca_type = typename ca_type::lca_type;

        using mesh_interval_t = typename ca_type::lca_type::mesh_interval_t;

        using mesh_t = samurai::MeshIDArray<ca_type, mesh_id_t>;

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;
        std::size_t nb_cells(std::size_t level, mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;

        std::size_t max_level() const;
        std::size_t min_level() const;
        const lca_type& domain() const;
        const ca_type& get_union() const;
        bool is_periodic(std::size_t d) const;
        const std::array<bool, dim>& periodicity() const;

        void swap(Mesh_base& mesh) noexcept;

        template <typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;
        const interval_t&
        get_interval(std::size_t level, const interval_t& interval, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const;
        const interval_t& get_interval(std::size_t level, const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const;

        template <typename... T>
        index_t get_index(std::size_t level, value_t i, T... index) const;
        index_t get_index(std::size_t level, const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const;

        void to_stream(std::ostream& os) const;

      protected:

        using derived_type = D;

        Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic);
        Mesh_base(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level);
        Mesh_base(const samurai::Box<double, dim>& b,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  const std::array<bool, dim>& periodic);

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        mesh_t& cells();

      private:

        void construct_domain();
        void construct_union();
        void update_sub_mesh();
        void renumbering();

        lca_type m_domain;
        std::size_t m_min_level;
        std::size_t m_max_level;
        std::array<bool, dim> m_periodic;
        mesh_t m_cells;
        ca_type m_union;
    };

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b,
                                           std::size_t start_level,
                                           std::size_t min_level,
                                           std::size_t max_level)
        : m_domain{start_level, b}
        , m_min_level{min_level}
        , m_max_level{max_level}
    {
        assert(min_level <= max_level);
        m_periodic.fill(false);
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b};

        construct_domain();
        construct_union();
        update_sub_mesh();
        renumbering();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b,
                                           std::size_t start_level,
                                           std::size_t min_level,
                                           std::size_t max_level,
                                           const std::array<bool, dim>& periodic)
        : m_domain{start_level, b}
        , m_min_level{min_level}
        , m_max_level{max_level}
        , m_periodic{periodic}
    {
        assert(min_level <= max_level);
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b};

        construct_domain();
        construct_union();
        update_sub_mesh();
        renumbering();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : m_min_level{min_level}
        , m_max_level{max_level}
    {
        assert(min_level <= max_level);
        m_periodic.fill(false);

        m_cells[mesh_id_t::cells] = {cl, false};

        construct_domain();
        construct_union();
        update_sub_mesh();
        renumbering();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic)
        : m_min_level{min_level}
        , m_max_level{max_level}
        , m_periodic{periodic}
    {
        assert(min_level <= max_level);
        m_cells[mesh_id_t::cells] = {cl, false};

        construct_domain();
        construct_union();
        update_sub_mesh();
        renumbering();
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::cells() -> mesh_t&
    {
        return m_cells;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(mesh_id_t mesh_id) const
    {
        return m_cells[mesh_id].nb_cells();
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(std::size_t level, mesh_id_t mesh_id) const
    {
        return m_cells[mesh_id][level].nb_cells();
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) const -> const ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::max_level() const
    {
        return m_max_level;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::min_level() const
    {
        return m_min_level;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::domain() const -> const lca_type&
    {
        return m_domain;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::get_union() const -> const ca_type&
    {
        return m_union;
    }

    template <class D, class Config>
    template <typename... T>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level, const interval_t& interval, T... index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index...);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level,
                                                   const interval_t& interval,
                                                   const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>& index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level, const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const
        -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, coord);
    }

    template <class D, class Config>
    template <typename... T>
    inline auto Mesh_base<D, Config>::get_index(std::size_t level, value_t i, T... index) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, i, index...);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::get_index(std::size_t level, const xt::xtensor_fixed<value_t, xt::xshape<dim>>& coord) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, coord);
    }

    template <class D, class Config>
    inline bool Mesh_base<D, Config>::is_periodic(std::size_t d) const
    {
        return m_periodic[d];
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::periodicity() const -> const std::array<bool, dim>&
    {
        return m_periodic;
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::swap(Mesh_base<D, Config>& mesh) noexcept
    {
        using std::swap;
        swap(m_cells, mesh.m_cells);
        swap(m_domain, mesh.m_domain);
        swap(m_union, mesh.m_union);
        swap(m_max_level, mesh.m_max_level);
        swap(m_min_level, mesh.m_min_level);
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_sub_mesh()
    {
        this->derived_cast().update_sub_mesh_impl();
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::renumbering()
    {
        m_cells[mesh_id_t::reference].update_index();

        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            if (mt != mesh_id_t::reference)
            {
                for (std::size_t level = 0; level <= max_refinement_level; ++level)
                {
                    lca_type& lhs       = m_cells[mt][level];
                    const lca_type& rhs = m_cells[mesh_id_t::reference][level];

                    auto expr = intersection(lhs, rhs);
                    expr.apply_interval_index(
                        [&](const auto& interval_index)
                        {
                            lhs[0][interval_index[0]].index = rhs[0][interval_index[1]].index;
                        });
                }
            }
        }
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_domain()
    {
        // lcl_type lcl = {m_cells[mesh_id_t::cells].max_level()};
        lcl_type lcl = {m_max_level};

        for_each_interval(m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              std::size_t shift = m_max_level - level;
                              interval_t to_add = i << shift;
                              auto shift_index  = index << shift;
                              static_nested_loop<dim - 1>(0,
                                                          1 << shift,
                                                          1,
                                                          [&](auto stencil)
                                                          {
                                                              auto new_index = shift_index + stencil;
                                                              lcl[new_index].add_interval(to_add);
                                                          });
                          });
        m_domain = {lcl};
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_union()
    {
        std::size_t min_lvl = m_cells[mesh_id_t::cells].min_level();
        std::size_t max_lvl = m_cells[mesh_id_t::cells].max_level();

        // FIX: cppcheck false positive ?
        // cppcheck-suppress redundantAssignment
        m_union[max_lvl] = m_cells[mesh_id_t::cells][max_lvl];
        for (std::size_t level = max_lvl - 1; level >= ((min_lvl == 0) ? 1 : min_lvl); --level)
        // for (std::size_t level = max_level - 1; level--> 0; )
        {
            lcl_type lcl{level};
            auto expr = union_(this->m_cells[mesh_id_t::cells][level], m_union[level + 1]).on(level);

            // std::cout << this->m_cells[mesh_id_t::cells][level] << std::endl;
            // std::cout << m_union[level+1] << std::endl;
            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            m_union[level] = {lcl};
        }
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::to_stream(std::ostream& os) const
    {
        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            os << fmt::format(fmt::emphasis::bold, "{}\n{:─^50}", mt, "") << std::endl;
            os << m_cells[id];
        }
    }

    template <class D, class Config>
    inline bool operator==(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
    {
        using mesh_id_t = typename Mesh_base<D, Config>::mesh_id_t;

        if (mesh1.max_level() != mesh2.max_level() || mesh1.min_level() != mesh2.min_level())
        {
            return false;
        }

        for (std::size_t level = mesh1.min_level(); level <= mesh1.max_level(); ++level)
        {
            if (!(mesh1[mesh_id_t::cells][level] == mesh2[mesh_id_t::cells][level]))
            {
                return false;
            }
        }
        return true;
    }

    template <class D, class Config>
    inline bool operator!=(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
    {
        return !(mesh1 == mesh2);
    }

    template <class D, class Config>
    inline std::ostream& operator<<(std::ostream& out, const Mesh_base<D, Config>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
} // namespace samurai
