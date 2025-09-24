// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>

#include "box.hpp"
#include "level_cell_array.hpp"
#include "level_cell_list.hpp"
#include "mesh.hpp"
#include "mesh_config.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    enum class UniformMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        count            = 2,
        reference        = cells_and_ghosts
    };

    template <std::size_t dim_, int ghost_width_ = default_config::ghost_width, class TInterval = default_config::interval_t>
    struct UniformConfig
    {
        static constexpr std::size_t dim = dim_;
        static constexpr int ghost_width = ghost_width_;
        using interval_t                 = TInterval;
        using mesh_id_t                  = UniformMeshId;
    };

    template <class Config>
    class UniformMesh
    {
      public:

        using config = Config;

        static constexpr std::size_t dim = config::dim;

        using mesh_id_t     = typename config::mesh_id_t;
        using interval_t    = typename config::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;

        using cl_type  = LevelCellList<dim, interval_t>;
        using ca_type  = LevelCellArray<dim, interval_t>;
        using coords_t = typename ca_type::coords_t;

        using mesh_t = MeshIDArray<ca_type, mesh_id_t>;

        UniformMesh() = default;
        UniformMesh(const cl_type& cl);
        UniformMesh(const ca_type& ca);
        UniformMesh(const Box<double, dim>& b,
                    std::size_t level,
                    double approx_box_tol = ca_type::default_approx_box_tol,
                    double scaling_factor = 0);

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;

        void swap(UniformMesh& mesh) noexcept;

        template <typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;

        template <class T1, typename... T>
        std::size_t get_index(T1 i, T... index) const;

        void to_stream(std::ostream& os) const;

        auto& origin_point() const;
        void set_origin_point(const coords_t& origin_point);
        auto scaling_factor() const;
        void set_scaling_factor(double scaling_factor);
        double cell_length(std::size_t level) const;

        auto cfg() const;

      private:

        void update_sub_mesh();
        void renumbering();

        mesh_t m_cells;
    };

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const Box<double, dim>& b, std::size_t level, double approx_box_tol, double scaling_factor_)
    {
        this->m_cells[mesh_id_t::cells] = {level, b, approx_box_tol, scaling_factor_};

        update_sub_mesh();
        renumbering();

        set_origin_point(origin_point());
        set_scaling_factor(scaling_factor());
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const cl_type& cl)
    {
        m_cells[mesh_id_t::cells] = {cl, false};

        update_sub_mesh();
        renumbering();

        set_origin_point(cl.origin_point());
        set_scaling_factor(cl.scaling_factor());
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const ca_type& ca)
    {
        m_cells[mesh_id_t::cells] = ca;

        update_sub_mesh();
        renumbering();

        set_origin_point(ca.origin_point());
        set_scaling_factor(ca.scaling_factor());
    }

    template <class Config>
    inline std::size_t UniformMesh<Config>::nb_cells(mesh_id_t mesh_id) const
    {
        return m_cells[mesh_id].nb_cells();
    }

    template <class Config>
    inline auto UniformMesh<Config>::operator[](mesh_id_t mesh_id) const -> const ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class Config>
    template <typename... T>
    inline auto UniformMesh<Config>::get_interval(std::size_t, const interval_t& interval, T... index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(interval, index...);
    }

    template <class Config>
    template <class T1, typename... T>
    inline std::size_t UniformMesh<Config>::get_index(T1 i, T... index) const
    {
        auto interval = m_cells[mesh_id_t::reference].get_interval(interval_t{i, i + 1}, index...);
        return interval.index + i;
    }

    template <class Config>
    inline void UniformMesh<Config>::swap(UniformMesh<Config>& mesh) noexcept
    {
        using std::swap;
        swap(m_cells, mesh.m_cells);
    }

    template <class Config>
    inline void UniformMesh<Config>::update_sub_mesh()
    {
        cl_type cl{this->m_cells[mesh_id_t::cells].level()};
        for_each_interval(this->m_cells[mesh_id_t::cells],
                          [&](std::size_t, const auto& interval, const auto& index_yz)
                          {
                              static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      cl[index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
                                  });
                          });

        m_cells[mesh_id_t::cells_and_ghosts] = {cl};
    }

    template <class Config>
    inline void UniformMesh<Config>::renumbering()
    {
        m_cells[mesh_id_t::reference].update_index();

        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            if (mt != mesh_id_t::reference)
            {
                for_each_interval(m_cells[mt],
                                  [&](std::size_t, auto& i, auto& index)
                                  {
                                      i.index = m_cells[mesh_id_t::reference].get_interval(i, index).index;
                                  });
            }
        }
    }

    template <class Config>
    inline void UniformMesh<Config>::to_stream(std::ostream& os) const
    {
        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            os << fmt::format(fmt::emphasis::bold, "{}\n{:─^50}", mt, "") << std::endl;
            os << m_cells[id];
        }
    }

    template <class Config>
    inline auto& UniformMesh<Config>::origin_point() const
    {
        return m_cells[0].origin_point();
    }

    template <class Config>
    inline void UniformMesh<Config>::set_origin_point(const coords_t& origin_point)
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(mesh_id_t::count); ++i)
        {
            m_cells[i].set_origin_point(origin_point);
        }
    }

    template <class Config>
    inline auto UniformMesh<Config>::scaling_factor() const
    {
        return m_cells[0].scaling_factor();
    }

    template <class Config>
    inline void UniformMesh<Config>::set_scaling_factor(double scaling_factor)
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(mesh_id_t::count); ++i)
        {
            m_cells[i].set_scaling_factor(scaling_factor);
        }
    }

    template <class Config>
    inline double UniformMesh<Config>::cell_length(std::size_t level) const
    {
        return samurai::cell_length(scaling_factor(), level);
    }

    template <class Config>
    inline auto UniformMesh<Config>::cfg() const
    {
        return mesh_config();
    }

    template <class Config>
    inline bool operator==(const UniformMesh<Config>& mesh1, const UniformMesh<Config>& mesh2)
    {
        using mesh_id_t = typename UniformMesh<Config>::mesh_id_t;

        return (mesh1[mesh_id_t::cells] == mesh2[mesh_id_t::cells]);
    }

    template <class Config>
    inline std::ostream& operator<<(std::ostream& out, const UniformMesh<Config>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
} // namespace samurai

template <>
struct fmt::formatter<samurai::UniformMeshId> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(samurai::UniformMeshId c, FormatContext& ctx) const
    {
        string_view name = "unknown";
        switch (c)
        {
            case samurai::UniformMeshId::cells:
                name = "cells";
                break;
            case samurai::UniformMeshId::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case samurai::UniformMeshId::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
