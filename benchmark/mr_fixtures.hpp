// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// MR-adapted meshes refined around the surfaces of balls, the copy of their
// sets into bench::SiteInputs, and their partitions into subdomains. Needs
// the mesh, the fields, the boundary conditions and the MR adaptation: only
// generate_set_algebra_inputs and bench_set_library_ops include it.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <xtensor/generators/xbuilder.hpp>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/domain_builder.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/sfc/hilbert.hpp>
#include <samurai/mpi/subdomain_bbox.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil.hpp>

#include "set_algebra_backend.hpp"
#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
#include "site_inputs.hpp"

namespace bench
{
    // disable_minimal_ghost_width keeps the ghost width at stencil_radius: by
    // default the mesh raises the max stencil radius to 2 (see SET_ALGEBRA.md).
    template <std::size_t dim>
    auto mr_config(std::size_t max_level, int stencil_radius)
    {
        return samurai::mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(stencil_radius).disable_minimal_ghost_width();
    }

    template <std::size_t dim>
    using mr_mesh_t = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), mr_config<dim>(4, 1)));

    template <std::size_t dim>
    using mr_field_t = decltype(samurai::make_scalar_field<double>("u", std::declval<mr_mesh_t<dim>&>()));

    // The mesh adapted to this indicator is refined to the finest level along
    // every ball surface. `shift` moves the balls along the diagonal.
    template <std::size_t dim>
    auto balls_indicator(std::size_t nballs, double shift)
    {
        return [centers = ball_centers<dim>(nballs, 0.25, 0.75),
                radius  = 0.2 / std::pow(static_cast<double>(nballs), 1. / static_cast<double>(dim)),
                shift](const auto& x)
        {
            for (const auto& c : centers)
            {
                double r2 = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const double dx = x[d] - c[d] - shift;
                    r2 += dx * dx;
                }
                if (r2 <= radius * radius)
                {
                    return 1.;
                }
            }
            return 0.;
        };
    }

    constexpr double mr_epsilon = 1e-3;

    // How far the fronts move in one time step.
    template <class Mesh>
    double step_shift(const Mesh& mesh)
    {
        return 4. * mesh.min_cell_length();
    }

    // The mesh of c before adaptation, on its geometry: the unit box, the
    // unit box with a hole as demos/FiniteVolume/linear_convection_obstacle
    // builds it, or the periodic unit box.
    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(const MeshCase& c)
    {
        auto config = mr_config<dim>(c.max_level, c.stencil_radius);
        const samurai::Box<double, dim> unit_box(xt::zeros<double>({dim}), xt::ones<double>({dim}));
        switch (c.geometry)
        {
            case Geometry::box:
                break;
            case Geometry::periodic:
                config.periodic(true);
                break;
            case Geometry::hole:
            {
                samurai::DomainBuilder<dim> domain(unit_box);
                domain.remove(samurai::Box<double, dim>(xt::ones<double>({dim}) * 0.5, xt::ones<double>({dim}) * 0.75));
                return samurai::mra::make_mesh(domain, config);
            }
        }
        return samurai::mra::make_mesh(unit_box, config);
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(const MeshCase& c, bool moved)
            : mesh(make_unit_mesh<dim>(c))
            , u(samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(c.nballs, moved ? step_shift(mesh) : 0.)))
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
            samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(mr_epsilon));
        }

        AdaptedCase(const AdaptedCase&)            = delete;
        AdaptedCase& operator=(const AdaptedCase&) = delete;
    };

    // Built once per binary and shared by its benchmarks. `moved` gives the
    // mesh of the next time step.
    template <std::size_t dim>
    AdaptedCase<dim>& adapted_case(const MeshCase& c, bool moved = false)
    {
        static std::map<std::tuple<mesh_key_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{c.key(), moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(c, moved);
        }
        return *entry;
    }

    template <class Mesh>
    void report_mesh(benchmark::State& state, const Mesh& mesh)
    {
        report_mesh(state, mesh[Mesh::mesh_id_t::cells]);
    }

    template <std::size_t dim>
    SiteInputs<dim> make_site_inputs(const mr_mesh_t<dim>& mesh)
    {
        using mesh_t    = mr_mesh_t<dim>;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        static_assert(std::is_same_v<typename mesh_t::ca_type, typename SiteInputs<dim>::ca_t>);
        static_assert(mesh_t::config_t::prediction_stencil_radius == prediction_stencil_radius);

        SiteInputs<dim> in;
        in.min_level          = mesh.min_level();
        in.max_level          = mesh.max_level();
        in.graduation_width   = mesh.graduation_width();
        in.max_stencil_radius = mesh.max_stencil_radius();
        in.domain_is_box      = mesh.domain().is_box();
        in.periodic           = mesh.periodicity();
        in.cells              = mesh[mesh_id_t::cells];
        in.cells_and_ghosts   = mesh[mesh_id_t::cells_and_ghosts];
        in.proj_cells         = mesh[mesh_id_t::proj_cells];
        in.reference          = mesh[mesh_id_t::reference];
        in.union_             = mesh.get_union();
        in.domain_pyramid     = mesh.domain_pyramid();
        for (std::size_t level = 0; level <= mesh.max_level(); ++level)
        {
            in.subdomain_pyramid[level] = mesh.subdomain(level);
        }
        samurai::for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                in.corners.push_back(mesh.corner(direction));
            });
        set_domain_bounds(in);
        return in;
    }

    // What Adapt::harten (mr/adapt.hpp) passes to make_graduation at the first
    // iteration of the adaptation of `adapted` to the fronts shifted by
    // `shift`: update_cell_array_from_tag(cells, tag), before graduation, in
    // in.ungraded_cells, and the cells it adds and removes
    // (cells_changed_by_tag, algorithm/graduation.hpp). A
    // copy of harten up to that call, for the configuration of AdaptedCase:
    // sequential, no relative detail, boundary not kept refined.
    template <std::size_t dim>
    void add_next_step(SiteInputs<dim>& in, const mr_mesh_t<dim>& adapted, std::size_t nballs, double shift)
    {
        using mesh_id_t = typename mr_mesh_t<dim>::mesh_id_t;

        auto mesh = adapted;
        auto u    = samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(nballs, shift));
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto detail = samurai::make_scalar_field<double>("detail", mesh, 0.);
        auto tag    = samurai::make_scalar_field<std::uint8_t>("tag", mesh, std::uint8_t{0});

        const std::size_t min_level = mesh.min_level();
        const std::size_t max_level = mesh.max_level();
        const auto cfg              = samurai::mra_config().epsilon(mr_epsilon);

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   tag[cell] = static_cast<std::uint8_t>(samurai::CellFlag::keep);
                               });
        samurai::update_ghost_mr(u);

        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < max_level; ++level)
        {
            samurai::ghosts_below_cells(mesh, level).apply_op(samurai::compute_detail(detail, u));
        }

        for (std::size_t level = std::max<std::size_t>(min_level, 1); level <= max_level; ++level)
        {
            const double eps_l = cfg.epsilon() / static_cast<double>(1 << (dim * (max_level - level)));
            auto subset_1      = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            subset_1.apply_op(samurai::mr_criteria(detail, tag, eps_l, cfg.regularity() + dim));
            samurai::update_tag_subdomains(level, tag, true);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            samurai::update_tag_periodic(level, tag);
            samurai::update_tag_subdomains(level, tag);
            keep_subset.apply_op(samurai::maximum(tag));
        }

        in.ungraded_cells = samurai::update_cell_array_from_tag(mesh[mesh_id_t::cells], tag);
        auto changes      = samurai::cells_changed_by_tag(mesh[mesh_id_t::cells], tag);
        in.tag_add_m      = std::move(changes.add_m);
        in.tag_remove_m   = std::move(changes.remove_m);
        in.tag_add_p      = std::move(changes.add_p);
        in.tag_remove_p   = std::move(changes.remove_p);
    }

    //////////////////////////////////////////////////////////////////////
    // Partitions
    //////////////////////////////////////////////////////////////////////

    // The cells of `cells` cut into `ranks` pieces of consecutive cells along
    // the Hilbert curve of the domain, of equal number of cells: what the SFC
    // load balancer (load_balancing/strategies/sfc.hpp) computes with one
    // atom per cell and a unit weight. The keys are those of sfc_cell_key, at
    // max_level, in the box of the domain.
    template <std::size_t dim>
    std::vector<samurai::CellArray<dim>> hilbert_pieces(const SiteInputs<dim>& whole, std::size_t ranks)
    {
        using index_t = typename samurai::LevelCellList<dim>::index_yz_t;

        struct Atom
        {
            samurai::load_balancing::sfc_key_t key;
            std::size_t level;
            int x;
            index_t yz;
        };

        const samurai::load_balancing::Hilbert curve;
        xt::xtensor_fixed<std::int64_t, xt::xshape<dim>> extent;
        for (std::size_t d = 0; d < dim; ++d)
        {
            extent[d] = whole.domain_max[d] - whole.domain_min[d];
        }
        std::vector<Atom> atoms;
        bench::for_each_interval(
            whole.cells,
            [&](std::size_t level, const auto& i, const auto& index)
            {
                const std::size_t shift = whole.max_level - level;
                for (int x = i.start; x < i.end; ++x)
                {
                    xt::xtensor_fixed<std::uint32_t, xt::xshape<dim>> p;
                    p(0) = static_cast<std::uint32_t>((static_cast<std::int64_t>(x) << shift) - whole.domain_min[0]);
                    for (std::size_t d = 1; d < dim; ++d)
                    {
                        p(d) = static_cast<std::uint32_t>((static_cast<std::int64_t>(index[d - 1]) << shift) - whole.domain_min[d]);
                    }
                    atoms.push_back({curve.template key<dim>(p, extent), level, x, index});
                }
            });
        std::stable_sort(atoms.begin(),
                         atoms.end(),
                         [](const Atom& a, const Atom& b)
                         {
                             return a.key < b.key;
                         });

        std::vector<samurai::CellList<dim>> lists(ranks);
        for (std::size_t k = 0; k < atoms.size(); ++k)
        {
            const auto& atom = atoms[k];
            lists[k * ranks / atoms.size()][atom.level][atom.yz].add_point(atom.x);
        }
        std::vector<samurai::CellArray<dim>> pieces;
        for (const auto& cl : lists)
        {
            pieces.emplace_back(cl);
        }
        return pieces;
    }

    // Mesh_base::find_neighbourhood (mesh.hpp:1309-1413), phases 1 and 2:
    // the ranks whose bounding box, expanded by the ghost reach of the pair,
    // meets that of `rank`, then those that meet it across a periodic
    // boundary (add_periodic_candidates, mesh.hpp:1418-1452), in increasing
    // order.
    template <std::size_t dim>
    std::vector<std::size_t> candidate_ranks(const std::vector<SiteInputs<dim>>& ranks, std::size_t rank)
    {
        using bbox_t        = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;
        const auto bounding = [&](std::size_t r)
        {
            auto bbox        = samurai::mpi_neighbor::compute_subdomain_bbox(ranks[r].subdomain());
            bbox.rank        = static_cast<int>(r);
            bbox.ghost_reach = sites::ghost_reach(ranks[r]) * ranks[r].subdomain().cell_length();
            return bbox;
        };
        std::vector<bbox_t> all_bboxes;
        for (std::size_t r = 0; r < ranks.size(); ++r)
        {
            all_bboxes.push_back(bounding(r));
        }
        const auto& my_bbox = all_bboxes[rank];

        std::set<std::size_t> candidates;
        for (std::size_t r = 0; r < ranks.size(); ++r)
        {
            if (r != rank && my_bbox.could_be_neighbor(all_bboxes[r]))
            {
                candidates.insert(r);
            }
        }
        if (sites::is_periodic(ranks[rank]))
        {
            std::array<int, dim> ones;
            ones.fill(1);
            const auto directions  = sites::periodic_directions(ones, 0, ranks[rank].periodic);
            const auto& domain     = ranks[rank].domain();
            const auto domain_size = xt::eval(domain.max_corner() - domain.min_corner());
            for (const auto& other_bbox : all_bboxes)
            {
                const auto other = static_cast<std::size_t>(other_bbox.rank);
                if (other == rank || candidates.contains(other))
                {
                    continue;
                }
                for (const auto& direction : directions)
                {
                    bbox_t shifted_bbox = other_bbox;
                    shifted_bbox.bbox.min_corner() += direction * domain_size;
                    shifted_bbox.bbox.max_corner() += direction * domain_size;
                    if (my_bbox.could_be_neighbor(shifted_bbox))
                    {
                        candidates.insert(other);
                        break;
                    }
                }
            }
        }
        return {candidates.begin(), candidates.end()};
    }

    // A CellArray of the level cell arrays of `levels`, the level of each
    // being its index.
    template <std::size_t dim>
    samurai::CellArray<dim> to_cell_array(const std::vector<samurai::LevelCellArray<dim>>& levels)
    {
        samurai::CellArray<dim> ca;
        for (const auto& lca : levels)
        {
            if (!lca.empty())
            {
                ca[lca.level()] = lca;
            }
        }
        return ca;
    }

    // The meshes the ranks of the partition c build, as MPI would build them
    // from the cells of the adapted mesh cut along the Hilbert curve
    // (hilbert_pieces) with Mesh_base(cells, reference mesh), which keeps the
    // domain of the reference mesh. The library cannot build them on one
    // process: the sets come from the sites, run with the static backend
    // (construct_subdomain, construct_union, find_neighbourhood, the scheme
    // ghosts and mesh_sub_mesh with the neighbours). The ranks have no tags
    // and no cells of a next step.
    template <std::size_t dim>
    std::vector<SiteInputs<dim>> make_partition(const MeshCase& c)
    {
        using S          = SamuraiStatic;
        const auto whole = make_site_inputs<dim>(adapted_case<dim>(c.whole()).mesh);

        std::vector<SiteInputs<dim>> ranks;
        for (auto& cells : hilbert_pieces(whole, c.ranks))
        {
            SiteInputs<dim> in;
            in.min_level          = whole.min_level;
            in.max_level          = whole.max_level;
            in.graduation_width   = whole.graduation_width;
            in.max_stencil_radius = whole.max_stencil_radius;
            in.domain_is_box      = whole.domain_is_box;
            in.periodic           = whole.periodic;
            in.domain_min         = whole.domain_min;
            in.domain_max         = whole.domain_max;
            in.domain_pyramid     = whole.domain_pyramid;
            in.corners            = whole.corners;
            in.cells              = std::move(cells);
            sites::construct_subdomain<S>(in,
                                          [&](std::size_t level, const auto& lca)
                                          {
                                              in.subdomain_pyramid[level] = lca;
                                          });
            sites::construct_union<S>(in,
                                      [&](std::size_t level, const auto& lca)
                                      {
                                          in.union_[level] = lca;
                                      });
            in.cells_and_ghosts = to_cell_array<dim>(sites::scheme_ghosts<S>(in));
            ranks.push_back(std::move(in));
        }

        for (std::size_t r = 0; r < ranks.size(); ++r)
        {
            ranks[r].candidates = candidate_ranks(ranks, r);
            ranks[r].neighbours = sites::find_neighbourhood<S>(ranks,
                                                               r,
                                                               [](std::size_t, const auto& set)
                                                               {
                                                                   return S::empty(set) ? std::size_t{0} : std::size_t{1};
                                                               });
        }

        std::vector<std::vector<samurai::LevelCellArray<dim>>> sub_meshes;
        for (std::size_t r = 0; r < ranks.size(); ++r)
        {
            std::vector<samurai::LevelCellArray<dim>> levels;
            sites::mesh_sub_mesh<S>(ranks[r],
                                    sites::neighbours_of(ranks, r),
                                    [&](std::size_t, const auto& lca)
                                    {
                                        levels.push_back(lca);
                                    });
            sub_meshes.push_back(std::move(levels));
        }
        const std::size_t n = whole.max_level + 2;
        for (std::size_t r = 0; r < ranks.size(); ++r)
        {
            const auto& levels        = sub_meshes[r];
            ranks[r].cells_and_ghosts = to_cell_array<dim>({levels.begin(), levels.begin() + static_cast<std::ptrdiff_t>(n)});
            ranks[r].reference        = to_cell_array<dim>(
                {levels.begin() + static_cast<std::ptrdiff_t>(n), levels.begin() + static_cast<std::ptrdiff_t>(2 * n)});
            ranks[r].proj_cells = to_cell_array<dim>({levels.begin() + static_cast<std::ptrdiff_t>(2 * n), levels.end()});
        }
        return ranks;
    }
} // namespace bench
