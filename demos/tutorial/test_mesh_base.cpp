// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <chrono>
#include <fmt/format.h>

#include <xtensor/xfixed.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/cell_flag.hpp>

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer                          = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

enum class SimpleID
{
    cells           = 0,
    ghost_and_cells = 1,
    count           = 2,
    reference       = ghost_and_cells
};

template <>
struct fmt::formatter<SimpleID> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(SimpleID c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case SimpleID::cells:
                name = "cells";
                break;
            case SimpleID::ghost_and_cells:
                name = "ghost_and_cells";
                break;
            case SimpleID::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

template <std::size_t dim_>
struct MyConfig
{
    static constexpr std::size_t dim                  = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    static constexpr std::size_t ghost_width          = 1;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t  = SimpleID;
};

template <class Config>
class MyMesh : public samurai::Mesh_base<MyMesh<Config>, Config>
{
  public:

    using base_type                  = samurai::Mesh_base<MyMesh<Config>, Config>;
    using config                     = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type   = typename base_type::cl_type;
    using lcl_type  = typename base_type::lcl_type;

    MyMesh(const MyMesh&)            = default;
    MyMesh& operator=(const MyMesh&) = default;

    MyMesh(MyMesh&&)            = default;
    MyMesh& operator=(MyMesh&&) = default;

    inline MyMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    inline MyMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b, start_level, min_level, max_level)
    {
    }

    void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              constexpr int gw = static_cast<int>(config::ghost_width);
                              lcl_type& lcl    = cl[level];
                              samurai::static_nested_loop<dim - 1, -gw, gw + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - gw, interval.end + gw});
                                  });
                          });
        this->m_cells[mesh_id_t::ghost_and_cells] = {cl, false};
    }
};

template <class Mesh>
bool check_overlap(const Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    bool overlap = false;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    for (std::size_t level1 = min_level; level1 < max_level; ++level1)
    {
        for (std::size_t level2 = level1 + 1; level2 <= max_level; ++level2)
        {
            auto subset = samurai::intersection(mesh[mesh_id_t::cells][level1], mesh[mesh_id_t::cells][level2]);
            subset(
                [&](const auto& interval, const auto& index)
                {
                    overlap = true;
                    std::cout << "overlap found between level " << level1 << " and level " << level2 << " at " << interval << " "
                              << index[0] << std::endl;
                });
        }
    }
    return overlap;
}

int main()
{
    constexpr std::size_t nb_bubbles  = 50;
    constexpr std::size_t dim         = 2;
    constexpr std::size_t start_level = 3;

    using container_t = xt::xtensor_fixed<double, xt::xshape<nb_bubbles>>;
    using mesh_t      = MyMesh<MyConfig<dim>>;

    container_t bb_xcenter, bb_ycenter;
    container_t bb0_xcenter, bb0_ycenter;
    container_t bb_radius;
    container_t dy;
    container_t omega;
    container_t aa;

    bb_xcenter  = 0.8 * xt::random::rand<double>({nb_bubbles}) + 0.1;
    bb0_xcenter = bb_xcenter;
    bb_ycenter  = xt::random::rand<double>({nb_bubbles}) - 0.5;
    bb0_ycenter = bb_ycenter;
    bb_radius   = 0.1 * xt::random::rand<double>({nb_bubbles}) + 0.02;
    dy          = 0.005 + 0.05 * xt::random::rand<double>({nb_bubbles});
    omega       = 0.5 * xt::random::rand<double>({nb_bubbles});
    aa          = 0.05 * xt::random::rand<double>({nb_bubbles});

    samurai::Box<double, dim> box({0, 0}, {1, 3});
    mesh_t mesh(box, start_level, 0, 9);

    std::cout << mesh << std::endl;
    using cl_type = typename mesh_t::cl_type;

    double t0 = 0;
    double t  = t0;
    double Dt = 0.5;

    for (std::size_t nite = 0; nite < 200; ++nite)
    {
        t += Dt;
        bb_xcenter = bb0_xcenter + aa * xt::cos(omega * t);
        bb_ycenter = bb_ycenter + Dt * dy;

        std::cout << "iteration " << nite << std::endl;

        tic();
        while (true)
        {
            if (check_overlap(mesh))
            {
                std::cout << mesh << std::endl;
                std::stringstream str;
                str << "bubble_2d_overlap_" << nite;
                samurai::save(str.str().data(), mesh);
                return 0;
            };

            auto tag        = samurai::make_field<int, 1>("tag", mesh);
            auto level_data = samurai::make_field<std::size_t, 1>("level", mesh);
            samurai::for_each_cell(mesh[SimpleID::cells],
                                   [&](auto cell)
                                   {
                                       level_data[cell] = cell.level;
                                   });

            samurai::for_each_cell(
                mesh[SimpleID::cells],
                [&](auto cell)
                {
                    bool inside    = false;
                    std::size_t ib = 0;

                    tag[cell] = static_cast<int>(samurai::CellFlag::keep);

                    while (!inside && ib < nb_bubbles)
                    {
                        double xc     = bb_xcenter[ib];
                        double yc     = bb_ycenter[ib];
                        double radius = bb_radius[ib];

                        auto corner = cell.corner();
                        auto center = cell.center();
                        double dx   = cell.length;

                        for (std::size_t i = 0; i < 2; ++i)
                        {
                            double x = corner[0] + static_cast<double>(i) * dx;
                            for (std::size_t j = 0; j < 2; ++j)
                            {
                                double y = corner[1] + static_cast<double>(j) * dx;
                                if (((!inside)
                                     && (std::pow((x - xc), 2.0) + pow((y - yc), 2.0) <= 1.25 * std::pow(radius, 2.0)
                                         && std::pow((x - xc), 2.0) + pow((y - yc), 2.0) >= 0.75 * std::pow(radius, 2.0)))
                                    || ((!inside)
                                        && (std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0) <= 1.25 * std::pow(radius, 2.0)
                                            && std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0)
                                                   >= 0.75 * std::pow(radius, 2.0))))
                                {
                                    if (cell.level < 9)
                                    {
                                        tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                                    }
                                    inside = true;
                                }
                            }
                        }
                        ib++;
                    }

                    if (cell.level > 0 && !inside)
                    {
                        tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
                    }
                });

            for (std::size_t level = mesh.max_level(); level >= 1; --level)
            {
                auto ghost_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::reference][level - 1]).on(level - 1);
                ghost_subset(
                    [&](const auto& i, const auto& index)
                    {
                        auto j = index[0];
                        tag(level - 1, i, j) |= static_cast<int>(samurai::CellFlag::keep);
                    });

                auto subset_2 = intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]);

                subset_2(
                    [&](const auto& interval, const auto& index)
                    {
                        auto i                    = interval;
                        auto j                    = index[0];
                        xt::xtensor<bool, 1> mask = (tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine));

                        for (int jj = -1; jj < 2; ++jj)
                        {
                            for (int ii = -1; ii < 2; ++ii)
                            {
                                xt::masked_view(tag(level, i + ii, j + jj), mask) |= static_cast<int>(samurai::CellFlag::keep);
                            }
                        }
                    });

                auto keep_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]).on(level - 1);
                keep_subset(
                    [&](const auto& interval, const auto& index)
                    {
                        auto i = interval;
                        auto j = index[0];

                        xt::xtensor<bool, 1> mask = (tag(level, 2 * i, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                                  | (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                                  | (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep))
                                                  | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep));

                        xt::masked_view(tag(level, 2 * i, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    });

                // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1},
                // {-1, -1}, {-1, 1}, {1, -1}};
                xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{
                    {1,  0 },
                    {-1, 0 },
                    {0,  1 },
                    {0,  -1}
                };

                for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto subset = samurai::intersection(samurai::translate(mesh[SimpleID::cells][level], s), mesh[SimpleID::cells][level - 1])
                                      .on(level);

                    subset(
                        [&](const auto& interval, const auto& index)
                        {
                            auto j_f = index[0];
                            auto i_f = interval.even_elements();

                            if (i_f.is_valid())
                            {
                                auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                                auto i_c  = i_f >> 1;
                                auto j_c  = j_f >> 1;
                                xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                                mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                                xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                                // tag(level - 1, i_c, j_c) |=
                                // static_cast<int>(samurai::CellFlag::keep);
                            }

                            i_f = interval.odd_elements();
                            if (i_f.is_valid())
                            {
                                auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                                auto i_c  = i_f >> 1;
                                auto j_c  = j_f >> 1;
                                xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                                mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                                xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                                // tag(level - 1, i_c, j_c) |=
                                // static_cast<int>(samurai::CellFlag::keep);
                            }
                        });
                }
            }

            cl_type cell_list;

            samurai::for_each_interval(mesh[SimpleID::cells],
                                       [&](std::size_t level, const auto& interval, const auto& index_yz)
                                       {
                                           std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
                                           for (int i = interval.start; i < interval.end; ++i)
                                           {
                                               if (tag[itag] & static_cast<int>(samurai::CellFlag::refine))
                                               {
                                                   samurai::static_nested_loop<dim - 1, 0, 2>(
                                                       [&](auto stencil)
                                                       {
                                                           auto index = 2 * index_yz + stencil;
                                                           cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                                       });
                                               }
                                               else if (tag[itag] & static_cast<int>(samurai::CellFlag::keep))
                                               {
                                                   cell_list[level][index_yz].add_point(i);
                                               }
                                               else
                                               {
                                                   cell_list[level - 1][index_yz >> 1].add_point(i >> 1);
                                               }
                                               itag++;
                                           }
                                       });

            mesh_t new_mesh(cell_list, mesh.min_level(), mesh.max_level());
            if (new_mesh == mesh)
            {
                break;
            }

            mesh = new_mesh;
        }
        auto duration = toc();
        std::cout << "adapt time: " << duration << std::endl;

        std::stringstream str;
        str << "bubble_2d_" << nite;
        samurai::save(str.str().data(), mesh);
    }
}