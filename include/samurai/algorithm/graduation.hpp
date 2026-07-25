// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xmasked_view.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/vector.hpp>

#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
namespace mpi = boost::mpi;
#endif

#include "../cell_flag.hpp"
#include "../concepts.hpp"
#include "../mesh.hpp"
#include "../stencil.hpp"
#include "../subset/node.hpp"
#include "../subset/utils.hpp"
#include "../timers.hpp"
#include "utils.hpp"

namespace samurai
{
    namespace detail
    {
        template <class T>
        SAMURAI_INLINE T start_shift_neg(T value, T shift)
        {
            return shift >= 0 ? value >> shift : value << -shift;
        }

        template <std::size_t d, class Translation, std::size_t dim>
        void get_periodic_directions(const Translation& translation,
                                     int delta,
                                     std::array<bool, dim> is_periodic,
                                     std::vector<DirectionVector<dim>>& directions,
                                     DirectionVector<dim>& current)
        {
            auto next = [&]()
            {
                if constexpr (d == dim - 1)
                {
                    directions.push_back(current);
                }
                else
                {
                    get_periodic_directions<d + 1>(translation, delta, is_periodic, directions, current);
                }
            };

            if (is_periodic[d])
            {
                current[d] = start_shift_neg(-translation[d], delta);
                next();

                current[d] = start_shift_neg(translation[d], delta);
                next();
            }
            current[d] = 0;
            next();
        }

        template <class Translation, std::size_t dim>
        auto get_periodic_directions(const Translation& translation, int delta, const std::array<bool, dim>& is_periodic)
        {
            DirectionVector<dim> current{};
            std::vector<DirectionVector<dim>> directions;
            get_periodic_directions<0>(translation, delta, is_periodic, directions, current);
            directions.pop_back();
            return directions;
        }
    }

    ///////////////////////
    // graduate operator //
    ///////////////////////

    template <std::size_t dim, class TInterval>
    class graduate_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(graduate_op)

        template <std::size_t d, class T, class Stencil>
        SAMURAI_INLINE void operator()(Dim<d>, T& tag, const Stencil& s) const
        {
            using namespace xt::placeholders;

            auto tag_func = [&](auto& i_f)
            {
                auto mask = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<std::uint8_t>(CellFlag::refine);
                auto i_c  = i_f >> 1;
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<std::uint8_t>(CellFlag::refine);
                                });

                auto mask2 = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<std::uint8_t>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask2,
                                [](auto& e)
                                {
                                    e |= static_cast<std::uint8_t>(CellFlag::keep);
                                });
            };

            if (auto i_even = i.even_elements(); i_even.is_valid())
            {
                tag_func(i_even);
            }

            if (auto i_odd = i.odd_elements(); i_odd.is_valid())
            {
                tag_func(i_odd);
            }
        }
    };

    template <class T, class Stencil>
    SAMURAI_INLINE auto graduate(T& tag, const Stencil& s)
    {
        return make_field_operator_function<graduate_op>(tag, s);
    }

    template <class Tag, class Stencil>
    void graduation(Tag& tag, const Stencil& stencil)
    {
        ScopedTimer timer("mesh graduation");
        auto& mesh      = tag.mesh();
        using mesh_t    = typename Tag::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level > 0; --level)
        {
            /**
             *
             *        |-----|-----| |-----|-----|
             *                                    --------------->
             *                                                             K
             *        |===========|-----------| |===========|-----------|
             */

            auto ghost_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);

            ghost_subset.apply_op(tag_to_keep<0>(tag));

            /**
             *                 R                                 K     R     K
             *        |-----|-----|=====|   ---------------> |-----|-----|=====|
             *
             */

            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            auto ghost_width = mesh.cfg().graduation_width();
            assert(ghost_width < 10 && "Graduation not implemented for ghost_width higher than 10");
            // maximum ghost width is set to 9
            static_for<1, 10>::apply(
                [&](auto static_ghost_width_)
                {
                    static constexpr int static_ghost_width = static_cast<int>(static_ghost_width_());
                    if (ghost_width == static_ghost_width)
                    {
                        subset_2.apply_op(tag_to_keep<static_ghost_width>(tag, CellFlag::refine));
                    }
                });

            /**
             *      K     C                          K     K
             *   |-----|-----|   -------------->  |-----|-----|
             *
             *   |-----------|
             *
             */

            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level - 1);
            keep_subset.apply_op(keep_children_together(tag));

            /**
             * Case 1
             * ======
             *                   R     K R     K
             *                |-----|-----|   --------------> |-----|-----| C or
             * K                                                 R
             *   |-----------| |-----------|
             *
             * Case 2
             * ======
             *                   K     K K     K
             *                |-----|-----|   --------------> |-----|-----| C K
             *   |-----------| |-----------|
             *
             */
            assert(stencil.shape()[1] == Tag::dim);
            for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
            {
                auto s      = xt::view(stencil, i);
                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1]).on(level);
                subset.apply_op(graduate(tag, s));
            }
        }
    }

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

    // Exchange a SINGLE level-array with the MPI neighbours (one message per neighbour),
    // returning their level-arrays. Used by the single-pass graduation to share, at each
    // level of its top-down sweep, only that level's cells - instead of serializing the
    // whole multi-level mesh. The message tag is the level so the per-level exchanges
    // (issued in lock-step across ranks) never cross.
    template <size_t dim, typename TInterval>
    std::vector<LevelCellArray<dim, TInterval>> exchange_level_mpi([[maybe_unused]] const LevelCellArray<dim, TInterval>& lca,
                                                                   [[maybe_unused]] const auto& mpi_neighbourhood,
                                                                   [[maybe_unused]] const int tag)
    {
        std::vector<LevelCellArray<dim, TInterval>> neighbour_levels(mpi_neighbourhood.size());
#ifdef SAMURAI_WITH_MPI
        if (mpi_neighbourhood.empty())
        {
            return neighbour_levels;
        }
        mpi::communicator world;
        std::vector<mpi::request> req;

        boost::mpi::packed_oarchive::buffer_type buffer;
        boost::mpi::packed_oarchive oa(world, buffer);
        oa << lca;

        for (const auto& neighbour : mpi_neighbourhood)
        {
            req.push_back(world.isend(neighbour.rank, tag, buffer));
        }
        std::size_t index = 0;
        for (const auto& neighbour : mpi_neighbourhood)
        {
            world.recv(neighbour.rank, tag, neighbour_levels[index++]);
        }
        mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
        return neighbour_levels;
    }

    // Boundary-contiguity cells folded into the single-pass graduation (see
    // graded_closure_single_pass). Both helpers are PURE functions of the graded level-l
    // cells they are given: `fine_sources` is this rank's graded level-l cells followed by
    // each neighbour's, so the result depends only on the UNION of the boundary cells and is
    // therefore INDEPENDENT of how the MPI partition splits them between ranks (the caller
    // then clips the result to the local coverage to keep only the cells it owns). This is
    // the single-pass analogue of list_interval_to_refine_for_contiguous_boundary_cells.

    // Case 1 (jump l -> l-1): the level-l cells sitting on each non-periodic physical face,
    // thickened n_contig cells inward (fine-cell units) so a coarser cell's wide stencil
    // never reaches past the domain. Returns the level-l cells to add.
    template <size_t dim, typename TInterval>
    LevelCellArray<dim, TInterval> boundary_case1_cells(size_t l,
                                                        int n_contig,
                                                        const CellArray<dim, TInterval>& domain,
                                                        const std::array<bool, dim>& is_periodic,
                                                        const std::vector<LevelCellArray<dim, TInterval>>& fine_sources)
    {
        using lca_t = LevelCellArray<dim, TInterval>;
        lca_t extra;
        bool have = false;
        for (const auto& fine_l : fine_sources)
        {
            for_each_cartesian_direction<dim>(
                [&](const auto dir, const auto& t)
                {
                    if (is_periodic[dir])
                    {
                        return;
                    }
                    const lca_t boundary_cells(difference(fine_l, translate(domain[l], -t)).on(l));
                    for (int i = 2; i <= n_contig; i += 2)
                    {
                        lca_t inside(intersection(translate(boundary_cells, -i * t), domain[l]).on(l));
                        extra = have ? lca_t(union_(extra, inside).on(l)) : std::move(inside);
                        have  = true;
                    }
                });
        }
        return extra;
    }

    // Case 2 (jump l-1 -> l, max_stencil_radius > 2): a coarse (l-1) face cell with a fine
    // cell within (radius-1) COARSE cells inward is refined to level l, pulling the fine
    // level out to the physical boundary. Expressed at l-1 (coarse domain face from the
    // pyramid + fine cells projected down) because the distance unit is the coarse cell.
    // Returns the level-l cells to add.
    template <size_t dim, typename TInterval>
    LevelCellArray<dim, TInterval> boundary_case2_cells(size_t l,
                                                        int max_stencil_radius,
                                                        const CellArray<dim, TInterval>& domain,
                                                        const std::array<bool, dim>& is_periodic,
                                                        const std::vector<LevelCellArray<dim, TInterval>>& fine_sources)
    {
        using lca_t = LevelCellArray<dim, TInterval>;
        lca_t extra;
        bool have = false;
        if (max_stencil_radius <= 2 || l == 0)
        {
            return extra;
        }
        const size_t lc = l - 1;
        for (const auto& fine_l : fine_sources)
        {
            const lca_t fine_on_c(self(fine_l).on(lc));
            for_each_cartesian_direction<dim>(
                [&](const auto dir, const auto& t)
                {
                    if (is_periodic[dir])
                    {
                        return;
                    }
                    const lca_t face_coarse(difference(domain[lc], translate(domain[lc], -t)).on(lc));
                    for (int i = 1; i < max_stencil_radius; ++i)
                    {
                        lca_t ref(self(lca_t(intersection(face_coarse, translate(fine_on_c, i * t)).on(lc))).on(l));
                        extra = have ? lca_t(union_(extra, ref).on(l)) : std::move(ref);
                        have  = true;
                    }
                });
        }
        return extra;
    }

    // if add the intervals in add_m_interval
    // if dim = 2 then add_m_interval stores the y coord
    // if dim > 2 then add_intercal contains the 'inner_stencil' i.e. the coordinates y+s_x, z+s_z, etc.
    // add_m_idx contains an array of indexes starting from 0 to add_m_interval.size()-1
    // lca_add_m is the destination
    template <size_t dim, typename TInterval, typename TCoord>
    void add_list_of_interval_back(const std::vector<TInterval>& intervals,  // in
                                   const TCoord& yz,                         // in, only used when dim == 2
                                   const std::vector<TCoord>& inner_stencil, // in
                                   std::vector<size_t>& idx,                 // inout
                                   LevelCellArray<dim, TInterval>& lca)      // out
    {
        assert(dim > 1); // cannot be static as the function will be defined (but not called) when dim=1
        if constexpr (dim > 2)
        {
            std::stable_sort(idx.begin(),
                             idx.end(),
                             [&inner_stencil](const size_t lhs_idx, const size_t rhs_idx) -> bool
                             {
                                 const auto& lhs = inner_stencil[lhs_idx];
                                 const auto& rhs = inner_stencil[rhs_idx];
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
                             });
        }
        TCoord outer_stencil = xt::xscalar(0);
        for (outer_stencil[dim - 2] = 0; outer_stencil[dim - 2] != 2; ++outer_stencil[dim - 2])
        {
            if constexpr (dim == 2)
            {
                for (size_t i = 0; i < intervals.size(); ++i)
                {
                    lca.add_interval_back(intervals[i], yz + outer_stencil);
                }
            }
            else
            {
                for (const size_t& i : idx)
                {
                    lca.add_interval_back(intervals[i], inner_stencil[i] + outer_stencil);
                }
            }
        } // end for
    }

    // Single-pass graded closure (no MPI neighbour, no periodicity, no boundary
    // contiguity - interior grading only). One top-down sweep, no fixed-point.
    //
    // F[l] = the region required at level >= l, as whole level-l cells, built from the
    // finest level down:
    //   F[max] = ca[max]
    //   F[l]   = ( ca[l] UNION expand(F[l+1].on(l), grad_width) ) rounded up to whole
    //            level-(l-1) parents
    // The rounding enforces samurai's complete-tree invariant: a cell of level l exists
    // iff its (l-1) parent is refined, which forces ALL that parent's children to level
    // >= l. So a coarse cell touched by grading is refined IN FULL - never partially,
    // which is what would leave a hole. A cell sits at exactly level l where it is
    // required at l but not at l+1:
    //   graded[l] = F[l] \ F[l+1].on(l)
    // Single-pass graded closure - interior grading only, one top-down sweep, no
    // fixed-point.
    //
    // F[l] = the region required at level >= l, as whole level-l cells, built from the
    // finest level down:
    //   F[max] = ca[max]
    //   F[l]   = ( ca[l] UNION expand( (F[l+1] U neighbours' F[l+1]).on(l), grad_width) )
    //            rounded up to whole level-(l-1) parents, clipped to the mesh coverage
    // Rounding to whole parents enforces samurai's complete-tree invariant (a coarse
    // cell touched by grading is refined IN FULL, never partially - which would leave a
    // hole). A cell sits at exactly level l where it is required at l but not at l+1:
    //   graded[l] = F[l] \ F[l+1].on(l)
    //
    // MPI: the sweep is synchronised across ranks over the GLOBAL level range (one tiny
    // all_reduce of the level bounds, once - not per iteration), and at each level only
    // that level's F is exchanged with the neighbours (exchange_level_mpi) - so a cascade
    // crossing a partition boundary is captured, without the whole-mesh exchange the
    // fixed-point loop does per iteration.
    // FoldBoundary is a COMPILE-TIME switch: when false the whole boundary-contiguity
    // machinery is `if constexpr`-eliminated, so the interior-only sweep (the common
    // max_stencil_radius == 1 case) pays exactly zero for it. make_graduation picks the
    // <false> instantiation for radius <= 1, so that case is byte-for-byte the interior sweep.
    template <std::size_t dim, class TInterval, size_t max_size, class MeshType, bool FoldBoundary = false>
    void graded_closure_single_pass(CellArray<dim, TInterval, max_size>& ca,
                                    const CellArray<dim, TInterval, max_size>& domain,
                                    const std::array<bool, dim>& is_periodic,
                                    const size_t grad_width,
                                    [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                                    [[maybe_unused]] const int max_stencil_radius = 1)
    {
        using ca_type = CellArray<dim, TInterval, max_size>;
        using lca_t   = typename ca_type::lca_type;

        const bool has_neighbour = !mpi_neighbourhood.empty();

        size_t max_level = ca.max_level();
        size_t min_level = ca.min_level();
#ifdef SAMURAI_WITH_MPI
        // Sweep over the GLOBAL level range so every rank issues the same per-level
        // exchanges in lock-step (an empty rank still participates with empty level
        // arrays). One tiny all_reduce (two ints) gives the TIGHT global range: this is
        // cheaper than the alternative of using the domain's full 0..max_level range,
        // which would make every rank exchange several empty coarse levels (extra
        // latency-bound messages per call). Gated on world.size() (identical on every
        // rank) so a neighbour-less rank still joins the collective and does not deadlock.
        {
            mpi::communicator world;
            if (world.size() > 1)
            {
                max_level = mpi::all_reduce(world, max_level, mpi::maximum<size_t>());
                min_level = mpi::all_reduce(world, min_level, mpi::minimum<size_t>());
            }
        }
#endif
        if (max_level <= min_level)
        {
            return; // a single level is always graded
        }
        const int w = static_cast<int>(grad_width);

        const bool any_periodic = std::any_of(is_periodic.begin(),
                                              is_periodic.end(),
                                              [](bool b)
                                              {
                                                  return b;
                                              });
        std::array<int, dim> nb_cells_finest_level{};
        if (any_periodic)
        {
            const auto& mn = domain[domain.max_level()].min_indices();
            const auto& mx = domain[domain.max_level()].max_indices();
            for (size_t d = 0; d != mx.size(); ++d)
            {
                nb_cells_finest_level[d] = mx[d] - mn[d];
            }
        }

        // Graduation only ever refines EXISTING cells, so every F[l] must be clipped to
        // the mesh coverage - otherwise the grading expansion would spill cells outside
        // the mesh. Without neighbours the mesh fills its domain, so the precomputed
        // domain pyramid domain[l] IS that coverage (cheap). With neighbours the local
        // mesh is only a subdomain, so we must clip to the LOCAL coverage (its own cells)
        // - clipping to the global domain would let a rank claim cells it does not own.
        const bool has_domain   = !domain.empty();
        const bool use_coverage = has_neighbour || !has_domain;
        lca_t coverage;
        if (use_coverage)
        {
            bool first = true;
            for (size_t l = ca.min_level(); l <= ca.max_level(); ++l)
            {
                if (ca[l].empty())
                {
                    continue;
                }
                lca_t projected(self(ca[l]).on(ca.max_level()));
                coverage = first ? projected : lca_t(union_(coverage, projected).on(ca.max_level()));
                first    = false;
            }
        }
        const auto clip = [&](lca_t&& x, size_t l) -> lca_t
        {
            return use_coverage ? lca_t(intersection(x, self(coverage).on(l)).on(l)) : lca_t(intersection(x, domain[l]).on(l));
        };

        std::array<lca_t, max_size> F;

        // Fold the boundary contiguity into the sweep at level l. The ENTIRE body is behind
        // `if constexpr (FoldBoundary)`, so for the interior-only instantiation (radius <= 1)
        // this lambda is empty and its per-level calls cost nothing - the common case pays
        // zero for the boundary machinery.
        //
        // When folding: case 2 runs FIRST (it can add boundary cells), then case 1 runs on
        // the UPDATED level-l cells, so it thickens the run case 2 just created - the
        // boundary->boundary feedback the looped algorithm needs a whole re-grade turn for.
        // Case 1 only ever adds cells inward (never new face cells), so a single case2 ->
        // case1 order converges with no within-level iteration. The driving level-l cells are
        // taken from THIS rank AND every neighbour (per-level graded-cell exchange, tags
        // disjoint from the F[l] exchange and from each other), so the result is independent
        // of the MPI partition.
        const auto extend_boundary = [&]([[maybe_unused]] size_t l)
        {
            if constexpr (FoldBoundary)
            {
                const int n_contig = std::max(max_stencil_radius, 2 * (max_stencil_radius - 2));
                if (max_stencil_radius <= 1 || domain.empty() || n_contig <= 1)
                {
                    return;
                }
                // The actual level-l cells (F[l] also carries everything required finer).
                const auto graded_level = [&](size_t ll) -> lca_t
                {
                    return (ll < max_level) ? lca_t(difference(F[ll], self(F[ll + 1]).on(ll)).on(ll)) : F[ll];
                };
                const auto exchange_graded_level = [&](const lca_t& graded_l, size_t ll, int phase) -> std::vector<lca_t>
                {
                    if (!has_neighbour)
                    {
                        return {};
                    }
                    return exchange_level_mpi(graded_l, mpi_neighbourhood, static_cast<int>(max_size * (1 + phase) + ll));
                };
                const auto grow_F = [&](size_t ll, lca_t&& piece)
                {
                    lca_t req(union_(F[ll], piece).on(ll));
                    F[ll] = (ll > min_level) ? clip(lca_t(self(req).on(ll - 1).on(ll)), ll) : clip(std::move(req), ll);
                };
                const auto sources = [&](const lca_t& my_fine, const std::vector<lca_t>& nbr_fine)
                {
                    std::vector<lca_t> s;
                    s.reserve(1 + nbr_fine.size());
                    s.push_back(my_fine);
                    s.insert(s.end(), nbr_fine.begin(), nbr_fine.end());
                    return s;
                };

                // Corner coupling: extending the run along one boundary face can create
                // cells that sit on a PERPENDICULAR face, which then need their own boundary
                // extension. Repeat case2+case1 until F[l] stops growing (bounded, local to
                // the corner region). Under MPI the continuation is agreed with one small
                // all_reduce per turn so every rank runs the same number of exchange turns
                // (a data-dependent local count would desynchronise the collective).
#ifdef SAMURAI_WITH_MPI
                mpi::communicator world;
#endif
                bool changed = true;
                while (
#ifdef SAMURAI_WITH_MPI
                    world.size() > 1 ? mpi::all_reduce(world, changed, std::logical_or()) : changed
#else
                    changed
#endif
                )
                {
                    const lca_t before = F[l];
                    lca_t g            = graded_level(l);
                    if (max_stencil_radius > 2 && l > 0)
                    {
                        grow_F(l,
                               boundary_case2_cells(l, max_stencil_radius, domain, is_periodic, sources(g, exchange_graded_level(g, l, 0))));
                        g = graded_level(l); // refresh: case 1 must see the cells case 2 just added
                    }
                    grow_F(l, boundary_case1_cells(l, n_contig, domain, is_periodic, sources(g, exchange_graded_level(g, l, 1))));
                    changed = !(F[l] == before);
                }
            }
        };

        std::array<std::vector<lca_t>, max_size> neighbour_F;
        F[max_level] = ca[max_level];
        extend_boundary(max_level);
        if (has_neighbour)
        {
            neighbour_F[max_level] = exchange_level_mpi(F[max_level], mpi_neighbourhood, static_cast<int>(max_level));
        }
        for (size_t l = max_level; l-- > min_level;)
        {
            // Finer requirement driving grading at level l: this rank's F[l+1] plus the
            // neighbours' F[l+1] (so a cascade from a neighbour reaches across the shared
            // boundary). The neighbour cells outside this subdomain are removed by the
            // coverage clip below; those within grad_width of the boundary force the local
            // boundary cells.
            lca_t finer = F[l + 1];
            for (const auto& nf : neighbour_F[l + 1])
            {
                // Union accumulation with a per-step .on() reconstruction: a raw loop reads better than std::accumulate.
                // cppcheck-suppress useStlAlgorithm
                finer = lca_t(union_(finer, nf).on(l + 1));
            }

            lca_t expanded(nestedExpand(self(finer).on(l), w));
            // Periodicity: a finer cell near a periodic boundary must also force
            // refinement near the opposite boundary (wrapped copies of the buffer).
            if (any_periodic)
            {
                const lca_t base = expanded;
                for (const auto& d : detail::get_periodic_directions(nb_cells_finest_level, int(domain.max_level()) - int(l), is_periodic))
                {
                    // Union accumulation with a per-step .on() reconstruction: a raw loop reads better than std::accumulate.
                    // cppcheck-suppress useStlAlgorithm
                    expanded = lca_t(union_(expanded, translate(base, d)).on(l));
                }
            }
            lca_t req(union_(ca[l], expanded).on(l));
            // Round up to whole (l-1) parents (complete-tree invariant) then clip.
            F[l] = (l > min_level) ? clip(lca_t(self(req).on(l - 1).on(l)), l) : clip(std::move(req), l);
            extend_boundary(l);
            if (has_neighbour)
            {
                neighbour_F[l] = exchange_level_mpi(F[l], mpi_neighbourhood, static_cast<int>(l));
            }
        }

        ca_type graded_ca;
        const auto collect = [&](size_t l)
        {
            return [&, l](const auto& x_interval, const auto& yz)
            {
                graded_ca[l].add_interval_back(x_interval, yz);
            };
        };
        self(F[max_level]).on(max_level)(collect(max_level));
        for (size_t l = min_level; l < max_level; ++l)
        {
            difference(F[l], self(F[l + 1]).on(l)).on(l)(collect(l));
        }
        std::swap(ca, graded_ca);
    }

    // Graduate `ca` in a SINGLE top-down sweep: interior 2:1 grading AND boundary contiguity
    // are folded into one pass (no outer fixed-point loop). At each level the boundary run is
    // extended BEFORE the next coarser level is graded, so the boundary->interior feedback is
    // captured within the same pass (see graded_closure_single_pass / extend_boundary). Both
    // boundary cases are handled - case 1 (jump l -> l-1, thicken the run inward) and case 2
    // (max_stencil_radius > 2, jump l-1 -> l, pull the fine level out to the boundary) - and
    // under MPI the boundary cells are gathered from the neighbours so the graded mesh is
    // independent of the partition. Returns 1 (the sweep is not iterative).
    //
    // Radius <= 1 (no boundary contiguity to enforce) dispatches to the <FoldBoundary=false>
    // instantiation: the plain interior sweep, with none of the boundary machinery compiled in.
    template <std::size_t dim, class TInterval, class MeshType, size_t max_size>
    size_t make_graduation(CellArray<dim, TInterval, max_size>& ca,
                           const CellArray<dim, TInterval, max_size>& domain,
                           const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                           const std::array<bool, dim>& is_periodic,
                           const size_t grad_width      = 1,
                           const int max_stencil_radius = 1 // half of width of the numerical scheme's stencil.
    )
    {
        ScopedTimer timer("make_graduation");
        if (max_stencil_radius <= 1)
        {
            graded_closure_single_pass<dim, TInterval, max_size, MeshType, false>(ca,
                                                                                  domain,
                                                                                  is_periodic,
                                                                                  grad_width,
                                                                                  mpi_neighbourhood,
                                                                                  max_stencil_radius);
        }
        else
        {
            graded_closure_single_pass<dim, TInterval, max_size, MeshType, true>(ca,
                                                                                 domain,
                                                                                 is_periodic,
                                                                                 grad_width,
                                                                                 mpi_neighbourhood,
                                                                                 max_stencil_radius);
        }
        return 1;
    }

    template <std::size_t dim, class TInterval, size_t max_size>
    size_t make_graduation(CellArray<dim, TInterval, max_size>& ca, const size_t grad_width = 1)
    {
        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> mpi_neighbourhood;
        std::array<bool, dim> is_periodic;
        CellArray<dim, TInterval, max_size> domain;

        is_periodic.fill(false);
        return make_graduation(ca, domain, mpi_neighbourhood, is_periodic, grad_width);
    }

    template <std::size_t dim, class TInterval, size_t max_size, class Tag>
    CellArray<dim, TInterval, max_size> update_cell_array_from_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
    {
        ScopedTimer timer("update_cell_array_from_tag");
        using size_type        = unsigned int;
        using value_t          = typename TInterval::value_t;
        using unsigned_value_t = typename std::make_unsigned_t<value_t>;
        using ca_type          = CellArray<dim, TInterval, max_size>;
        using coord_type       = typename ca_type::lca_type::coord_type;

        const auto& mesh = tag.mesh();

        // On an empty (sub)mesh min_level() returns max_size + 1 and max_level()
        // returns 0, so start_level > end_level: the half-open `<` test then runs
        // zero iterations instead of walking off the level array (a `!=` test
        // would overflow past max_size and segfault). An empty rank -- e.g. right
        // after load balancing concentrates every cell elsewhere -- must adapt to
        // an empty cell array, not crash.
        const size_t start_level = old_ca.min_level();
        const size_t end_level   = old_ca.max_level() + 1;

        // create the ensemble of cells to coarsen
        ca_type ca_add_m;
        ca_type ca_remove_m;
        ca_type ca_add_p;
        ca_type ca_remove_p;

        std::vector<TInterval> add_p_interval;
        std::vector<coord_type> add_p_inner_stencil;
        std::vector<size_t> add_p_idx;

        for (size_t level = start_level; level < end_level; ++level)
        {
            const auto begin = old_ca[level].cbegin();
            const auto end   = old_ca[level].cend();
            for (auto it = begin; it != end; ++it)
            {
                const auto& x_interval = *it;
                const auto& yz         = it.index();
                const bool is_yz_even  = dim == 1 or xt::all(xt::equal(yz % 2, 0));

                for (value_t x = x_interval.start; x < x_interval.end; ++x)
                {
                    const size_type itag         = static_cast<size_type>(x_interval.index) + static_cast<unsigned_value_t>(x);
                    const bool refine            = tag[itag] & static_cast<std::uint8_t>(CellFlag::refine);
                    const bool coarsenAndNotKeep = tag[itag] & static_cast<std::uint8_t>(CellFlag::coarsen)
                                               and not(tag[itag] & static_cast<std::uint8_t>(CellFlag::keep));
                    if (refine and level < mesh.max_level())
                    {
                        ca_remove_p[level].add_point_back(x, yz);
                        if constexpr (dim == 1)
                        {
                            ca_add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, {});
                        }
                        else
                        {
                            nestedLoop<dim - 1, 0, dim - 2>(0,
                                                            2,
                                                            [&](const auto& inner_stencil)
                                                            {
                                                                add_p_interval.push_back({2 * x, 2 * x + 2});
                                                                if constexpr (dim > 2)
                                                                {
                                                                    add_p_inner_stencil.emplace_back(2 * yz + inner_stencil);
                                                                    add_p_idx.push_back(add_p_interval.size() - 1); // std::iota
                                                                                                                    // on the fly
                                                                }
                                                            });
                        }
                    }
                    else if (coarsenAndNotKeep and level > mesh.min_level())
                    {
                        if (x % 2 == 0 and is_yz_even) // should be modified when using load balancing.
                        {
                            ca_add_m[level - 1].add_point_back(x >> 1, yz >> 1);
                        }
                        ca_remove_m[level].add_point_back(x, yz);
                    }
                } // end for each x
                if (dim != 1 and (it + 1 == end or (it + 1).index()[dim - 2] != yz[dim - 2]))
                {
                    add_list_of_interval_back(add_p_interval, coord_type(2 * yz), add_p_inner_stencil, add_p_idx, ca_add_p[level + 1]);
                    add_p_interval.clear();
                    add_p_inner_stencil.clear();
                    add_p_idx.clear();
                }
            } // end for each interval
        } // end for each level
        CellArray<dim, TInterval, max_size> new_ca;
        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            auto set = difference(union_(old_ca[level], ca_add_m[level], ca_add_p[level]), union_(ca_remove_m[level], ca_remove_p[level]));
            set(
                [&](const auto& x_interval, const auto& yz)
                {
                    new_ca[level].add_interval_back(x_interval, yz);
                });
        }
        return new_ca;
    }
}
