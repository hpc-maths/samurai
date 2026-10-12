// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The set expressions samurai evaluates, as the library writes them, on
// MR-adapted meshes in 1D, 2D and 3D, next to the library operations that
// evaluate them.
//
// Each BM_Site_* benchmark rebuilds, level by level, the expressions of one
// library function (copied from the source cited above it) and traverses
// them with a callback that only counts: its time is the cost of the set
// algebra alone in that function. Where the function applies a cheap field
// operator to the set, BM_SiteOp_* applies that operator instead, so that
// Site / SiteOp is the share of the set algebra in the function.
//
// BM_Op_* time the library operations themselves (ghost update, one
// adaptation step, mesh construction, graduation, field transfer, upwind
// flux). The adaptation step also reports the time of its phases, read from
// samurai's timers, in ms per iteration.
//
// Args: (max_level, number of balls). The mesh is refined to max_level along
// the surfaces of the balls (bench::adapted_case); more balls means more,
// smaller fronts at constant max_level.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/boundary.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/schemes/fv.hpp>

#include "set_algebra_fixtures.hpp"

namespace
{
    template <std::size_t dim>
    using mesh_t = bench::mr_mesh_t<dim>;

    template <std::size_t dim>
    using mesh_id_t = typename mesh_t<dim>::mesh_id_t;

    template <std::size_t dim>
    bench::AdaptedCase<dim>& get_case(benchmark::State& state, bool moved = false)
    {
        auto& c = bench::adapted_case<dim>(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)), moved);
        bench::report_mesh(state, c.mesh);
        return c;
    }

    template <std::size_t dim>
    std::array<samurai::DirectionVector<dim>, dim> positive_directions()
    {
        std::array<samurai::DirectionVector<dim>, dim> dirs;
        for (std::size_t d = 0; d < dim; ++d)
        {
            dirs[d].fill(0);
            dirs[d][d] = 1;
        }
        return dirs;
    }

    // Time spent in samurai's own timers during the benchmark loop, reported
    // in ms per iteration (the timers are only active when enabled).
    class PhaseTimers
    {
      public:

        explicit PhaseTimers(std::vector<std::string> names)
            : m_names(std::move(names))
        {
            samurai::times::timers.enable();
            for (const auto& name : m_names)
            {
                m_before.push_back(elapsed_us(name));
            }
        }

        void report(benchmark::State& state) const
        {
            for (std::size_t p = 0; p < m_names.size(); ++p)
            {
                state.counters[m_names[p] + " [ms]"] = (elapsed_us(m_names[p]) - m_before[p]) / 1000.
                                                     / static_cast<double>(state.iterations());
            }
        }

      private:

        static double elapsed_us(const std::string& name)
        {
            return static_cast<double>(samurai::times::timers.getElapsedTime(name).count());
        }

        std::vector<std::string> m_names;
        std::vector<double> m_before;
    };

    // Runs `site` with a callback that traverses each set and counts.
    template <class Site>
    void run_site(benchmark::State& state, Site&& site)
    {
        bench::Tally tally;
        std::size_t nsets = 0;
        for (auto _ : state)
        {
            tally = {};
            nsets = 0;
            site(
                [&](std::size_t, const auto& set)
                {
                    tally += bench::traverse(set);
                    ++nsets;
                });
        }
        bench::report(state, tally);
        state.counters["sets"] = static_cast<double>(nsets);
    }

    /////////////////////////////////////////////////////////////////////
    // The sites
    /////////////////////////////////////////////////////////////////////

    // Mesh_base::update_mesh_neighbour (mesh.hpp): the union of the cells of
    // all the finer levels, projected level by level.
    template <class Mesh, class F>
    void site_mesh_union(const Mesh& mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = mesh.max_level(); level >= 1; --level)
        {
            f(level - 1, samurai::union_(mesh[mesh_id_t::cells][level], mesh.get_union()[level]).on(level - 1));
        }
    }

    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp): the ghosts the prediction
    // needs one and two levels below each cell, then the projection cells.
    template <class Mesh, class F>
    void site_mesh_prediction_ghosts(const Mesh& mesh, F&& f)
    {
        using mesh_id_t       = typename Mesh::mesh_id_t;
        constexpr int pred_r  = Mesh::config_t::prediction_stencil_radius;
        const auto& cells     = mesh[mesh_id_t::cells];
        const auto& cag       = mesh[mesh_id_t::cells_and_ghosts];
        const auto& subdomain = mesh.subdomain();
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || cells[level].empty())
            {
                continue;
            }
            f(level - 2, samurai::nestedExpand(samurai::self(cells[level]).on(level - 2), pred_r));
            f(level - 1,
              samurai::intersection(samurai::nestedExpand(samurai::self(cag[level]).on(level - 1), pred_r),
                                    samurai::nestedExpand(samurai::self(subdomain).on(level - 1), pred_r)));
        }
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= mesh[mesh_id_t::reference].max_level(); ++level)
        {
            f(level, samurai::intersection(mesh[mesh_id_t::reference][level], mesh.get_union()[level]));
        }
    }

    // Mesh_base::construct_corners (mesh.hpp): the inner cells at each corner
    // and edge of the domain, at the finest level, for every diagonal
    // direction. Runs at every mesh construction although the domain does not
    // change.
    template <class Mesh, class F>
    void site_corners(const Mesh& mesh, F&& f)
    {
        constexpr std::size_t dim = Mesh::dim;
        if constexpr (dim > 1)
        {
            using direction_t      = samurai::DirectionVector<dim>;
            const auto& domain_lca = mesh.domain();
            samurai::for_each_diagonal_direction<dim>(
                [&](const auto& direction)
                {
                    std::array<direction_t, dim> shifts;
                    std::size_t n = 0;
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        if (direction[i] != 0)
                        {
                            shifts[n].fill(0);
                            shifts[n][i] = -direction[i];
                            ++n;
                        }
                    }
                    if (n == 2)
                    {
                        f(domain_lca.level(),
                          samurai::difference(
                              domain_lca,
                              samurai::union_(samurai::translate(domain_lca, shifts[0]), samurai::translate(domain_lca, shifts[1]))));
                    }
                    else if constexpr (dim > 2)
                    {
                        f(domain_lca.level(),
                          samurai::difference(domain_lca,
                                              samurai::union_(samurai::translate(domain_lca, shifts[0]),
                                                              samurai::translate(domain_lca, shifts[1]),
                                                              samurai::translate(domain_lca, shifts[2]))));
                    }
                });
        }
    }

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, bottom-up.
    template <class Mesh, class F>
    void site_ghost_projection(const Mesh& mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = mesh.max_level(); level >= 1; --level)
        {
            f(level - 1, samurai::intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1));
        }
    }

    // update_ghost_mr: prediction of the ghosts, top-down.
    template <class Mesh, class F>
    void site_ghost_prediction(const Mesh& mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = 1; level <= mesh.max_level(); ++level)
        {
            auto pred_ghosts = samurai::difference(mesh[mesh_id_t::all_cells][level],
                                                   samurai::union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            f(level, samurai::intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level));
        }
    }

    // update_outer_ghosts (algorithm/update_outer_ghost.hpp), without the
    // boundary condition itself: corner ghosts projected one and two levels
    // down (project_corner_below), boundary ghosts projected from the level
    // above (project_bc) and predicted to the level above (predict_bc), for
    // every level and Cartesian direction, with one layer of boundary ghosts
    // (Dirichlet). predict_bc builds a LevelCellArray
    // (domain_boundary_outer_layer) on every call.
    template <class Mesh, class F>
    void site_outer_ghosts(const Mesh& mesh, F&& f)
    {
        using mesh_id_t                = typename Mesh::mesh_id_t;
        constexpr std::size_t dim      = Mesh::dim;
        constexpr int n_bc_ghosts      = 1;
        const std::size_t lowest_level = mesh.min_level() > 0 ? mesh.min_level() - 1 : 0;
        for (std::size_t level = mesh.max_level() + 1; level-- > lowest_level;)
        {
            if constexpr (dim > 1)
            {
                if (level >= mesh.min_level())
                {
                    samurai::for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            for (std::size_t delta_l = 1; delta_l <= 2 && delta_l <= level; ++delta_l)
                            {
                                auto fine_inner_corner = samurai::self(mesh.corner(direction)).on(level);
                                auto fine_outer_corner = samurai::intersection(samurai::translate(fine_inner_corner, direction),
                                                                               mesh[mesh_id_t::reference][level]);
                                f(level - delta_l,
                                  samurai::intersection(fine_outer_corner.on(level - delta_l), mesh[mesh_id_t::reference][level - delta_l]));
                            }
                        });
                }
            }
            samurai::for_each_cartesian_direction<dim>(
                [&](auto, const auto& direction)
                {
                    if (level < mesh.max_level())
                    {
                        auto domain        = samurai::self(mesh.domain()).on(level);
                        auto outside_layer = samurai::difference(samurai::translate(mesh.get_union()[level], direction), domain);
                        f(level, samurai::intersection(outside_layer, mesh[mesh_id_t::reference][level]).on(level));
                    }
                    if (level < mesh.max_level() && level >= mesh.min_level())
                    {
                        auto bc_ghosts = samurai::domain_boundary_outer_layer(mesh, level, direction, n_bc_ghosts);
                        f(level + 1, samurai::intersection(bc_ghosts, mesh[mesh_id_t::reference][level + 1]).on(level + 1));
                    }
                });
        }
    }

    // site_outer_ghosts reading the domain at each level from the pyramid
    // the mesh already stores (mesh.domain(level)) instead of projecting the
    // finest domain with .on(level) each time: not what the library does
    // today, the gain of a cached coarse set.
    template <class Mesh, class F>
    void site_outer_ghosts_cached_domain(const Mesh& mesh, F&& f)
    {
        using mesh_id_t                = typename Mesh::mesh_id_t;
        constexpr std::size_t dim      = Mesh::dim;
        constexpr int n_bc_ghosts      = 1;
        const std::size_t lowest_level = mesh.min_level() > 0 ? mesh.min_level() - 1 : 0;
        for (std::size_t level = mesh.max_level() + 1; level-- > lowest_level;)
        {
            if constexpr (dim > 1)
            {
                if (level >= mesh.min_level())
                {
                    samurai::for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            for (std::size_t delta_l = 1; delta_l <= 2 && delta_l <= level; ++delta_l)
                            {
                                auto fine_inner_corner = samurai::self(mesh.corner(direction)).on(level);
                                auto fine_outer_corner = samurai::intersection(samurai::translate(fine_inner_corner, direction),
                                                                               mesh[mesh_id_t::reference][level]);
                                f(level - delta_l,
                                  samurai::intersection(fine_outer_corner.on(level - delta_l), mesh[mesh_id_t::reference][level - delta_l]));
                            }
                        });
                }
            }
            samurai::for_each_cartesian_direction<dim>(
                [&](auto, const auto& direction)
                {
                    if (level < mesh.max_level())
                    {
                        auto outside_layer = samurai::difference(samurai::translate(mesh.get_union()[level], direction), mesh.domain(level));
                        f(level, samurai::intersection(outside_layer, mesh[mesh_id_t::reference][level]).on(level));
                    }
                    if (level < mesh.max_level() && level >= mesh.min_level())
                    {
                        auto bc_ghosts = samurai::domain_boundary_outer_layer(mesh, level, direction, n_bc_ghosts);
                        f(level + 1, samurai::intersection(bc_ghosts, mesh[mesh_id_t::reference][level + 1]).on(level + 1));
                    }
                });
        }
    }

    // site_mesh_prediction_ghosts reading the subdomain from the pyramid the
    // mesh already stores (mesh.subdomain(level)): the gain of a cached coarse
    // set, not what the library does today.
    template <class Mesh, class F>
    void site_mesh_prediction_ghosts_cached_subdomain(const Mesh& mesh, F&& f)
    {
        using mesh_id_t      = typename Mesh::mesh_id_t;
        constexpr int pred_r = Mesh::config_t::prediction_stencil_radius;
        const auto& cells    = mesh[mesh_id_t::cells];
        const auto& cag      = mesh[mesh_id_t::cells_and_ghosts];
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || cells[level].empty())
            {
                continue;
            }
            f(level - 2, samurai::nestedExpand(samurai::self(cells[level]).on(level - 2), pred_r));
            f(level - 1,
              samurai::intersection(samurai::nestedExpand(samurai::self(cag[level]).on(level - 1), pred_r),
                                    samurai::nestedExpand(mesh.subdomain(level - 1), pred_r)));
        }
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= mesh[mesh_id_t::reference].max_level(); ++level)
        {
            f(level, samurai::intersection(mesh[mesh_id_t::reference][level], mesh.get_union()[level]));
        }
    }

    // Adapt::harten (mr/adapt.hpp): detail computation.
    template <class Mesh, class F>
    void site_detail(const Mesh& mesh, F&& f)
    {
        using mesh_id_t             = typename Mesh::mesh_id_t;
        const std::size_t min_level = mesh.min_level();
        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < mesh.max_level(); ++level)
        {
            f(level,
              samurai::intersection(mesh[mesh_id_t::all_cells][level],
                                    samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                  .on(level));
        }
    }

    // Adapt::harten: tagging (mr_criteria) and tag finalization (maximum),
    // which evaluate the same set.
    template <class Mesh, class F>
    void site_tag(const Mesh& mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = std::max<std::size_t>(mesh.min_level(), 1); level <= mesh.max_level(); ++level)
        {
            f(level - 1, samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1));
        }
    }

    // make_graduation on tags (algorithm/graduation.hpp, the AMR path): every
    // direction of the star stencil, every level.
    template <class Mesh, class F>
    void site_graduation_tag(const Mesh& mesh, F&& f)
    {
        using mesh_id_t           = typename Mesh::mesh_id_t;
        constexpr std::size_t dim = Mesh::dim;
        const auto stencil        = samurai::star_stencil<dim>();
        for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
        {
            for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
            {
                auto s = xt::view(stencil, i);
                f(level,
                  samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1]).on(level));
            }
        }
    }

    // for_each_interface (interface.hpp): the faces between two cells of the
    // same level, then between a cell and a finer one, in both orientations,
    // for every Cartesian direction.
    template <class Mesh, class F>
    void site_interfaces(const Mesh& mesh, F&& f)
    {
        using mesh_id_t           = typename Mesh::mesh_id_t;
        constexpr std::size_t dim = Mesh::dim;
        const auto& cells         = mesh[mesh_id_t::cells];
        for (const auto& direction : positive_directions<dim>())
        {
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                f(level, samurai::intersection(cells[level], samurai::translate(cells[level], -direction)));
                if (level < cells.max_level())
                {
                    f(level + 1, samurai::intersection(cells[level], samurai::translate(cells[level + 1], -direction)).on(level + 1));
                    f(level + 1, samurai::intersection(cells[level], samurai::translate(cells[level + 1], direction)).on(level + 1));
                }
            }
        }
    }

    // update_fields (algorithm/update_fields.hpp): copy of the unchanged
    // cells, then coarsening and refinement between the old and new meshes.
    template <class Mesh, class F>
    void site_field_transfer(const Mesh& mesh, const Mesh& new_mesh, F&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            f(level, samurai::intersection(mesh[mesh_id_t::reference][level], new_mesh[mesh_id_t::cells][level]));
        }
        for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
        {
            f(level - 1, samurai::intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1));
            f(level - 1, samurai::intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1));
        }
    }
} // namespace

/////////////////////////////////////////////////////////////////////////
// Set algebra alone
/////////////////////////////////////////////////////////////////////////

template <std::size_t dim>
static void BM_Site_MeshUnion(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_mesh_union(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_MeshPredictionGhosts(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_mesh_prediction_ghosts(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_Corners(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_corners(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_GhostProjection(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_ghost_projection(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_GhostPrediction(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_ghost_prediction(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_OuterGhosts(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_outer_ghosts(mesh, f);
             });
}

template <std::size_t dim>
static void BM_SiteCached_OuterGhosts(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_outer_ghosts_cached_domain(mesh, f);
             });
}

template <std::size_t dim>
static void BM_SiteCached_MeshPredictionGhosts(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_mesh_prediction_ghosts_cached_subdomain(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_Detail(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_detail(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_Tag(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_tag(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_GraduationTag(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_graduation_tag(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_Interfaces(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_interfaces(mesh, f);
             });
}

template <std::size_t dim>
static void BM_Site_FieldTransfer(benchmark::State& state)
{
    const auto& mesh     = get_case<dim>(state).mesh;
    const auto& new_mesh = get_case<dim>(state, true).mesh;
    run_site(state,
             [&](auto&& f)
             {
                 site_field_transfer(mesh, new_mesh, f);
             });
}

/////////////////////////////////////////////////////////////////////////
// The same sets with the field operator of the library
/////////////////////////////////////////////////////////////////////////

template <std::size_t dim>
static void BM_SiteOp_GhostProjection(benchmark::State& state)
{
    auto& u = get_case<dim>(state).u;
    for (auto _ : state)
    {
        site_ghost_projection(u.mesh(),
                              [&](std::size_t, const auto& set)
                              {
                                  set.apply_op(samurai::variadic_projection(u));
                              });
        benchmark::DoNotOptimize(u.array().data());
    }
}

template <std::size_t dim>
static void BM_SiteOp_GhostPrediction(benchmark::State& state)
{
    auto& u                          = get_case<dim>(state).u;
    constexpr std::size_t pred_order = mesh_t<dim>::config_t::prediction_stencil_radius;
    for (auto _ : state)
    {
        site_ghost_prediction(u.mesh(),
                              [&](std::size_t, const auto& set)
                              {
                                  set.apply_op(samurai::variadic_prediction<pred_order, false>(u));
                              });
        benchmark::DoNotOptimize(u.array().data());
    }
}

template <std::size_t dim>
static void BM_SiteOp_Detail(benchmark::State& state)
{
    auto& u     = get_case<dim>(state).u;
    auto detail = samurai::make_scalar_field<double>("detail", u.mesh());
    detail.fill(0);
    for (auto _ : state)
    {
        site_detail(u.mesh(),
                    [&](std::size_t, const auto& set)
                    {
                        set.apply_op(samurai::compute_detail(detail, u));
                    });
        benchmark::DoNotOptimize(detail.array().data());
    }
}

template <std::size_t dim>
static void BM_SiteOp_Tag(benchmark::State& state)
{
    auto& u     = get_case<dim>(state).u;
    auto& mesh  = u.mesh();
    auto detail = samurai::make_scalar_field<double>("detail", mesh);
    auto tag    = samurai::make_scalar_field<std::uint8_t>("tag", mesh);
    detail.fill(0);
    tag.fill(0);
    for (auto _ : state)
    {
        site_tag(mesh,
                 [&](std::size_t level, const auto& set)
                 {
                     const double eps_l = bench::mr_epsilon / static_cast<double>(1 << (dim * (mesh.max_level() - level - 1)));
                     set.apply_op(samurai::mr_criteria(detail, tag, eps_l, 1. + dim));
                 });
        benchmark::DoNotOptimize(tag.array().data());
    }
}

/////////////////////////////////////////////////////////////////////////
// Library operations
/////////////////////////////////////////////////////////////////////////

template <std::size_t dim>
static void BM_Op_UpdateGhost(benchmark::State& state)
{
    auto& u = get_case<dim>(state).u;
    for (auto _ : state)
    {
        samurai::update_ghost_mr(u);
        benchmark::DoNotOptimize(u.array().data());
    }
}

// One adaptation step: the field of the moved fronts, on the mesh adapted to
// the unmoved ones. The copies are not timed. Counters: time of the phases
// of the adaptation, in ms per step.
template <std::size_t dim>
static void BM_Op_AdaptStep(benchmark::State& state)
{
    const auto& c = get_case<dim>(state);
    const PhaseTimers timers(
        {"ghost update", "detail computation", "tag cells", "tag finalization", "make_graduation", "mesh construction", "fields update"});

    for (auto _ : state)
    {
        state.PauseTiming();
        auto mesh = c.mesh;
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        bench::init_balls<dim>(u, static_cast<std::size_t>(state.range(1)), bench::step_shift<dim>(mesh.max_level()));
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto adapt = samurai::make_MRAdapt(u);
        state.ResumeTiming();

        adapt(samurai::mra_config().epsilon(bench::mr_epsilon));
        benchmark::DoNotOptimize(u.array().data());
    }

    timers.report(state);
}

// MRMesh(cells, reference mesh): every derived mesh id, the union, the
// neighbourhood. Builds the mesh of the moved fronts.
template <std::size_t dim>
static void BM_Op_MeshConstruction(benchmark::State& state)
{
    const auto& mesh     = get_case<dim>(state).mesh;
    const auto& new_mesh = get_case<dim>(state, true).mesh;
    const auto& ca       = new_mesh[mesh_id_t<dim>::cells];
    const PhaseTimers timers({"update_sub_mesh", "construct_union", "update_mesh_neighbour", "construct_subdomain", "construct_corners"});
    for (auto _ : state)
    {
        mesh_t<dim> built(ca, mesh);
        benchmark::DoNotOptimize(built);
    }
    timers.report(state);
}

template <std::size_t dim>
static void BM_Op_Graduation(benchmark::State& state)
{
    const auto& mesh = get_case<dim>(state, true).mesh;
    for (auto _ : state)
    {
        state.PauseTiming();
        auto ca = mesh[mesh_id_t<dim>::cells];
        state.ResumeTiming();
        samurai::make_graduation(ca,
                                 mesh.domain_pyramid(),
                                 mesh.mpi_neighbourhood(),
                                 mesh.periodicity(),
                                 mesh.graduation_width(),
                                 mesh.max_stencil_radius());
        benchmark::DoNotOptimize(ca);
    }
}

// update_fields from the mesh of the fronts to the mesh of the moved fronts.
template <std::size_t dim>
static void BM_Op_FieldTransfer(benchmark::State& state)
{
    const auto& c        = get_case<dim>(state);
    const auto& new_mesh = get_case<dim>(state, true).mesh;
    for (auto _ : state)
    {
        state.PauseTiming();
        auto mesh = c.mesh;
        auto dest = new_mesh;
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        u.array() = c.u.array();
        state.ResumeTiming();
        samurai::update_fields(dest, u);
        benchmark::DoNotOptimize(u.array().data());
    }
}

// Upwind convection flux: the interfaces of section `site_interfaces` plus
// the flux computation itself.
template <std::size_t dim>
static void BM_Op_UpwindFlux(benchmark::State& state)
{
    auto& u = get_case<dim>(state).u;
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);
    auto conv = samurai::make_convection_upwind<std::decay_t<decltype(u)>>(velocity);
    samurai::update_ghost_mr(u);
    for (auto _ : state)
    {
        auto flux = conv(u);
        benchmark::DoNotOptimize(flux.array().data());
    }
}

/////////////////////////////////////////////////////////////////////////
// Registration
/////////////////////////////////////////////////////////////////////////

namespace
{
    template <std::size_t dim>
    void mesh_args(benchmark::internal::Benchmark* b)
    {
        const long max_level = static_cast<long>(bench::default_max_level<dim>());
        if constexpr (dim == 1)
        {
            // a 1D front is a point: many of them to get a non-trivial mesh
            b->Args({max_level, 16});
            b->Args({max_level, 256});
        }
        else
        {
            b->Args({max_level, 1});
            b->Args({max_level, (dim == 2) ? 16 : 8});
        }
        if constexpr (dim == 2)
        {
            b->Args({max_level + 2, 16});
        }
        b->Unit(benchmark::kMicrosecond);
    }
} // namespace

#define SAMURAI_BENCH_CALL_SITES(DIM)                                          \
    BENCHMARK(BM_Site_MeshUnion<DIM>)->Apply(mesh_args<DIM>);                  \
    BENCHMARK(BM_Site_MeshPredictionGhosts<DIM>)->Apply(mesh_args<DIM>);       \
    BENCHMARK(BM_SiteCached_MeshPredictionGhosts<DIM>)->Apply(mesh_args<DIM>); \
    BENCHMARK(BM_Site_Corners<DIM>)->Apply(mesh_args<DIM>);                    \
    BENCHMARK(BM_Op_MeshConstruction<DIM>)->Apply(mesh_args<DIM>);             \
    BENCHMARK(BM_Site_GhostProjection<DIM>)->Apply(mesh_args<DIM>);            \
    BENCHMARK(BM_SiteOp_GhostProjection<DIM>)->Apply(mesh_args<DIM>);          \
    BENCHMARK(BM_Site_GhostPrediction<DIM>)->Apply(mesh_args<DIM>);            \
    BENCHMARK(BM_SiteOp_GhostPrediction<DIM>)->Apply(mesh_args<DIM>);          \
    BENCHMARK(BM_Site_OuterGhosts<DIM>)->Apply(mesh_args<DIM>);                \
    BENCHMARK(BM_SiteCached_OuterGhosts<DIM>)->Apply(mesh_args<DIM>);          \
    BENCHMARK(BM_Op_UpdateGhost<DIM>)->Apply(mesh_args<DIM>);                  \
    BENCHMARK(BM_Site_Detail<DIM>)->Apply(mesh_args<DIM>);                     \
    BENCHMARK(BM_SiteOp_Detail<DIM>)->Apply(mesh_args<DIM>);                   \
    BENCHMARK(BM_Site_Tag<DIM>)->Apply(mesh_args<DIM>);                        \
    BENCHMARK(BM_SiteOp_Tag<DIM>)->Apply(mesh_args<DIM>);                      \
    BENCHMARK(BM_Site_GraduationTag<DIM>)->Apply(mesh_args<DIM>);              \
    BENCHMARK(BM_Op_Graduation<DIM>)->Apply(mesh_args<DIM>);                   \
    BENCHMARK(BM_Site_FieldTransfer<DIM>)->Apply(mesh_args<DIM>);              \
    BENCHMARK(BM_Op_FieldTransfer<DIM>)->Apply(mesh_args<DIM>);                \
    BENCHMARK(BM_Op_AdaptStep<DIM>)->Apply(mesh_args<DIM>);                    \
    BENCHMARK(BM_Site_Interfaces<DIM>)->Apply(mesh_args<DIM>);                 \
    BENCHMARK(BM_Op_UpwindFlux<DIM>)->Apply(mesh_args<DIM>);

SAMURAI_BENCH_CALL_SITES(1)
SAMURAI_BENCH_CALL_SITES(2)
SAMURAI_BENCH_CALL_SITES(3)

BENCHMARK_MAIN();
