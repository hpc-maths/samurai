// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <fmt/format.h>

#include "box.hpp"
#include "level_cell_array.hpp"
#include "level_cell_list.hpp"
#include "mesh.hpp"
#include "samurai_config.hpp"
#include "subset/subset_op.hpp"

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

        using cl_type = LevelCellList<dim, interval_t>;
        using ca_type = LevelCellArray<dim, interval_t>;

        using mesh_t = MeshIDArray<ca_type, mesh_id_t>;

        UniformMesh(const cl_type& cl);
        UniformMesh(const Box<double, dim>& b, std::size_t level);

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;

        void swap(UniformMesh& mesh) noexcept;

        template <typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;

        template <class T1, typename... T>
        std::size_t get_index(T1 i, T... index) const;

        void to_stream(std::ostream& os) const;

      private:

        void update_sub_mesh();
        void renumbering();

        mesh_t m_cells;
    };

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const Box<double, dim>& b, std::size_t level)
    {
        this->m_cells[mesh_id_t::cells] = {level, b};

        update_sub_mesh();
        renumbering();
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const cl_type& cl)
    {
        m_cells[mesh_id_t::cells] = {cl, false};

        update_sub_mesh();
        renumbering();
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
                ca_type& lhs       = m_cells[mt];
                const ca_type& rhs = m_cells[mesh_id_t::reference];

                auto expr = intersection(lhs, rhs);
                expr.apply_interval_index(
                    [&](const auto& interval_index)
                    {
                        lhs[0][interval_index[0]].index = rhs[0][interval_index[1]].index;
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
    auto format(samurai::UniformMeshId c, FormatContext& ctx)
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
