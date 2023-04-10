// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/mesh.hpp>

/**
 *
 * Cells:
 * ======
 *
 * level: 2                                         |--|--|--|--|
 *
 * level: 1                                   |-----|           |-----|-----|
 *
 * level: 0             |----------|----------| |----------|
 *
 *
 * Cells and ghosts:
 * =================
 *
 * level: 2                                      |==|--|--|--|--|==|
 *
 * level: 1 |=====|-----|=====|=====|-----|-----|=====|
 *
 * level: 0  |==========|----------|----------|
 * |==========|----------|==========|
 *
 */

enum class MeshID
{
    cells            = 0,               // Leaves (where the computation is done)
    cells_and_ghosts = 1,               // Leaves + ghosts
    count            = 2,               // Total number of cells categories
    reference        = cells_and_ghosts // Which is the largest ID including all the others
};

template <std::size_t dim_>
struct MeshConfig
{
    static constexpr std::size_t dim                  = dim_;
    static constexpr std::size_t max_refinement_level = 20;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t  = MeshID;
};

template <class Config>
class Mesh : public samurai::Mesh_base<Mesh<Config>, Config>
{
  public:

    // Importing all the types used in what follows
    using base_type                  = samurai::Mesh_base<Mesh<Config>, Config>;
    using config                     = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type   = typename base_type::cl_type;
    using lcl_type  = typename base_type::lcl_type;

    // Constructor starting from a cell list
    inline Mesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    // Constructor from a given box (domain)
    inline Mesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b, start_level, min_level, max_level)
    {
    }

    // This specifies how to add the ghosts once we know the leaves
    void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->cells()[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, auto)
                          {
                              lcl_type& lcl = cl[level];
                              lcl[{}].add_interval({interval.start - 1, interval.end + 1});
                          });
        // Put into the cells_and_ghosts category
        this->cells()[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};

template <>
struct fmt::formatter<MeshID> : formatter<string_view>
{
    template <typename FormatContext>
    auto format(MeshID c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case MeshID::cells:
                name = "cells";
                break;
            case MeshID::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case MeshID::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
