#pragma once

#include "algorithm.hpp"
#include "field.hpp"
#include "stencil.hpp"
#include "subset/subset_op.hpp"

namespace samurai
{
    template <class Mesh, std::size_t neighbourhood_width = 1>
    bool is_graduated(const Mesh& mesh, const Stencil<1 + 2 * Mesh::dim * neighbourhood_width, Mesh::dim> stencil = star_stencil<Mesh::dim>())
    {
        bool cond = true;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                {
                    auto s   = xt::view(stencil, is);
                    auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                    set(
                        [&cond](const auto&, const auto&)
                        {
                            cond = false;
                        });
                    if (!cond)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <class Mesh, std::size_t neighbourhood_width = 1>
    void make_graduation(Mesh& mesh, const Stencil<1 + 2 * Mesh::dim * neighbourhood_width, Mesh::dim> stencil = star_stencil<Mesh::dim>())
    {
        static constexpr std::size_t dim = Mesh::dim;
        using cl_type                    = typename Mesh::cl_type;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        auto tag = make_field<bool, 1>("tag", mesh);

        while (true)
        {
            tag.resize();
            tag.fill(false);

            for (std::size_t level = min_level + 2; level <= max_level; ++level)
            {
                for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
                {
                    for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                    {
                        auto s   = xt::view(stencil, is);
                        auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                        set(
                            [&](const auto& i, const auto& index)
                            {
                                tag(level_below, i, index) = true;
                            });
                    }
                }
            }

            cl_type cl;
            for_each_interval(mesh,
                              [&](std::size_t level, const auto& interval, const auto& index_yz)
                              {
                                  auto itag = interval.start + interval.index;
                                  for (auto i = interval.start; i < interval.end; ++i, ++itag)
                                  {
                                      if (tag[itag])
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](auto stencil)
                                              {
                                                  auto index = 2 * index_yz + stencil;
                                                  cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index_yz].add_point(i);
                                      }
                                  }
                              });
            Mesh new_ca = {cl, true};

            if (new_ca == mesh)
            {
                break;
            }

            std::swap(mesh, new_ca);
        }
    }
}
