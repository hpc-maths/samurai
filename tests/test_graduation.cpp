#include <algorithm>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(graduation, dim_1)
    {
        constexpr size_t dim = 1;
        CellList<dim> cl;
        cl[0][{}].add_interval({1, 2});
        cl[5][{}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_2)
    {
        constexpr size_t dim = 2;
        CellList<dim> cl;
        cl[0][{}].add_interval({1, 2});
        cl[5][{}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_3)
    {
        constexpr size_t dim = 3;
        CellList<dim> cl;
        cl[0][{1, 1}].add_interval({1, 2});
        cl[5][{0, 0}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    // Build the 1D domain pyramid [0, n) at `top_level`, projected down to level 0, as the
    // multi-level CellArray make_graduation expects (domain[l] = domain at level l).
    static CellArray<1> domain_pyramid_1d(int top_level, int n_cells_top)
    {
        CellList<1> cl;
        for (int l = top_level; l >= 0; --l)
        {
            const int n = n_cells_top >> (top_level - l);
            cl[static_cast<std::size_t>(l)][{}].add_interval({0, n});
        }
        return CellArray<1>{cl};
    }

    // Boundary contiguity exercised end to end through the production make_graduation (the
    // folded single pass). A level-4 run [14,16) touches the right physical boundary and a
    // level-3 cell [6,7) (fine [12,14)) sits one coarse cell inside it. For a wide stencil
    // that L3 cell must be refined to level 4 so the boundary run is thick enough. The mesh
    // tiles the domain (coverage == domain).
    static void expect_boundary_refined(int radius)
    {
        const std::array<bool, 1> is_periodic{false};

        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> no_neighbours;

        CellList<1> cl;
        cl[4][{}].add_interval({14, 16});
        cl[3][{}].add_interval({0, 7});
        CellArray<1> ca{cl};

        make_graduation(ca, domain_pyramid_1d(4, 16), no_neighbours, is_periodic, size_t{1}, radius);

        EXPECT_TRUE(is_graduated(ca));

        // The boundary-adjacent level-3 cell [6,7) must have been refined to level 4, so
        // nothing remains at level 3 there.
        CellList<1> hole_cl;
        hole_cl[3][{}].add_interval({6, 7});
        const LevelCellArray<1> hole = CellArray<1>{hole_cl}[3];
        size_t remaining             = 0;
        intersection(ca[3], hole)
            .on(3)(
                [&](const auto&, const auto&)
                {
                    ++remaining;
                });
        EXPECT_EQ(remaining, 0u) << "the boundary-adjacent level-3 cell must be refined (radius=" << radius << ")";
    }

    TEST(graduation, boundary_contiguity_radius2)
    {
        expect_boundary_refined(2);
    }

    TEST(graduation, boundary_contiguity_radius3)
    {
        expect_boundary_refined(3);
    }

    // Corner coupling (2D). At a domain corner two physical boundary faces meet, so the
    // boundary run extended along one face can create cells that then need extension along
    // the perpendicular face. This reproduces the exact mesh (a central level-5 block inside
    // a level-4 frame, radius 3) that exposed a fused single-pass under-refinement at the
    // corners: the fine level must be pulled into the corner, which needs the per-level
    // extension to be iterated until the corner region stabilises.
    TEST(graduation, boundary_contiguity_corner_2d)
    {
        constexpr size_t dim = 2;

        // domain pyramid: [0, 16)^2 at level 4 (= [0, 32)^2 at level 5), down to level 0.
        CellList<dim> domain_cl;
        for (int l = 5; l >= 0; --l)
        {
            const int n = 32 >> (5 - l);
            for (int j = 0; j < n; ++j)
            {
                domain_cl[static_cast<std::size_t>(l)][{j}].add_interval({0, n});
            }
        }
        CellArray<dim> domain{domain_cl};

        // Tiling input: central level-5 block [4,28)^2 (= [2,14)^2 at level 4) surrounded by
        // a level-4 frame filling [0,16)^2.
        CellList<dim> cl;
        for (int j = 0; j < 16; ++j) // level-4 rows
        {
            if (j < 2 || j >= 14)
            {
                cl[4][{j}].add_interval({0, 16});
            }
            else
            {
                cl[4][{j}].add_interval({0, 2});
                cl[4][{j}].add_interval({14, 16});
            }
        }
        for (int j = 4; j < 28; ++j) // level-5 rows of the central block
        {
            cl[5][{j}].add_interval({4, 28});
        }
        CellArray<dim> ca{cl};

        const std::array<bool, dim> is_periodic{false, false};

        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> no_neighbours;
        make_graduation(ca, domain, no_neighbours, is_periodic, size_t{1}, 3);

        EXPECT_TRUE(is_graduated(ca));

        // The fine level (5) must be pulled into the bottom-left corner: level 4 must NOT
        // remain at the very corner cell (x=[0,2)@L4, row 0). The buggy single pass left L4
        // there instead of refining to L5.
        CellList<dim> corner_cl;
        corner_cl[4][{0}].add_interval({0, 2});
        const LevelCellArray<dim> corner = CellArray<dim>{corner_cl}[4];
        size_t remaining                 = 0;
        intersection(ca[4], corner)
            .on(4)(
                [&](const auto&, const auto&)
                {
                    ++remaining;
                });
        EXPECT_EQ(remaining, 0u) << "the fine level must be pulled into the domain corner (corner coupling)";
    }

    static size_t count_intervals(size_t l, const LevelCellArray<1>& a)
    {
        size_t n = 0;
        self(a).on(l)(
            [&](const auto&, const auto&)
            {
                ++n;
            });
        return n;
    }

    static bool same_lca(size_t l, const LevelCellArray<1>& a, const LevelCellArray<1>& b)
    {
        return count_intervals(l, LevelCellArray<1>(difference(a, b).on(l))) == 0
            && count_intervals(l, LevelCellArray<1>(difference(b, a).on(l))) == 0;
    }

    // The fused single-pass boundary helpers (boundary_case1_cells / boundary_case2_cells)
    // must be INDEPENDENT of the MPI partition: a physical-boundary cell owned by a
    // NEIGHBOUR rank must drive exactly the same refinement as if this rank owned it. This
    // is the single-pass analogue of graduation.contiguous_boundary_partition_independent,
    // tested directly on the pure helpers (no real MPI needed) by feeding the boundary run
    // either as this rank's own cells or as a neighbour source.
    TEST(graduation, fused_boundary_case1_partition_independent)
    {
        CellList<1> domain_cl;
        domain_cl[4][{}].add_interval({0, 16});
        domain_cl[3][{}].add_interval({0, 8});
        CellArray<1> domain{domain_cl};
        const std::array<bool, 1> is_periodic{false};
        const int radius   = 3;
        const int n_contig = std::max(radius, 2 * (radius - 2)); // = 3

        CellList<1> bcl;
        bcl[4][{}].add_interval({14, 16}); // level-4 run on the right physical boundary
        const LevelCellArray<1> boundary = CellArray<1>{bcl}[4];
        const LevelCellArray<1> empty4;

        const LevelCellArray<1> all_local = boundary_case1_cells(size_t{4},
                                                                 n_contig,
                                                                 domain,
                                                                 is_periodic,
                                                                 std::vector<LevelCellArray<1>>{boundary});
        const LevelCellArray<1> split     = boundary_case1_cells(size_t{4},
                                                             n_contig,
                                                             domain,
                                                             is_periodic,
                                                             std::vector<LevelCellArray<1>>{empty4, boundary});

        EXPECT_GT(count_intervals(4, all_local), 0u) << "the boundary run must force an inner refinement";
        EXPECT_TRUE(same_lca(4, all_local, split)) << "case-1 refinement must not depend on which rank owns the boundary cell";
    }

    TEST(graduation, fused_boundary_case2_partition_independent)
    {
        CellList<1> domain_cl;
        domain_cl[4][{}].add_interval({0, 16});
        domain_cl[3][{}].add_interval({0, 8});
        CellArray<1> domain{domain_cl};
        const std::array<bool, 1> is_periodic{false};
        const int radius = 3;

        // A level-4 (fine) cell sitting one coarse cell inside the right boundary, so case 2
        // must pull the fine level out to the boundary. Provided either locally or by a neighbour.
        CellList<1> fcl;
        fcl[4][{}].add_interval({12, 14});
        const LevelCellArray<1> fine = CellArray<1>{fcl}[4];
        const LevelCellArray<1> empty4;

        const LevelCellArray<1> all_local = boundary_case2_cells(size_t{4}, radius, domain, is_periodic, std::vector<LevelCellArray<1>>{fine});
        const LevelCellArray<1> split = boundary_case2_cells(size_t{4},
                                                             radius,
                                                             domain,
                                                             is_periodic,
                                                             std::vector<LevelCellArray<1>>{empty4, fine});

        EXPECT_TRUE(same_lca(4, all_local, split)) << "case-2 refinement must not depend on which rank owns the fine cell";
    }

    namespace
    {
        // samurai::graduation() (as opposed to make_graduation() above) reads the graduation width
        // from mesh.cfg() and dispatches it through dispatch_static<1, 9>.
        //
        // The tag is built on holder(mesh) so that it owns its mesh: a plain field only points to
        // the mesh it was built on, which here is a local of this helper.
        auto make_graduation_width_test_tag(std::size_t graduation_width)
        {
            static constexpr std::size_t dim = 1;
            using box_t                      = Box<double, dim>;

            auto mesh_cfg = mesh_config<dim>().min_level(1).max_level(3).graduation_width(graduation_width);
            auto mesh     = mra::make_mesh(box_t{xt::zeros<double>({dim}), xt::ones<double>({dim})}, mesh_cfg);
            auto m        = holder(mesh);
            auto tag      = make_scalar_field<int>("tag", m);
            tag.fill(static_cast<int>(CellFlag::keep));
            return tag;
        }
    }

    // A graduation width of 0 means "no graduation constraint" and must not throw.
    TEST(graduation, width_zero_is_a_noop)
    {
        auto tag = make_graduation_width_test_tag(0);
        const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};
        EXPECT_NO_THROW(samurai::graduation(tag, stencil));
    }

    // A width beyond what dispatch_static<1, 9> is instantiated for must fail loudly
    // (std::out_of_range) instead of silently skipping the graduation step.
    TEST(graduation, width_above_dispatch_range_throws)
    {
        auto tag = make_graduation_width_test_tag(15);
        const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};
        EXPECT_THROW(samurai::graduation(tag, stencil), std::out_of_range);
    }

    namespace
    {
        // make_graduation written straight from its definition with plain set expressions,
        // without the shortcuts of the production code (no MPI neighbours). F[l] is the region
        // required at level >= l, built from the finest level down; graded[l] = F[l] \ F[l+1].
        template <std::size_t dim>
        CellArray<dim> reference_graduation(const CellArray<dim>& ca,
                                            const CellArray<dim>& domain,
                                            const std::array<bool, dim>& is_periodic,
                                            std::size_t grad_width,
                                            int radius)
        {
            using lca_t                 = LevelCellArray<dim>;
            const std::size_t min_level = ca.min_level();
            const std::size_t max_level = ca.max_level();
            if (max_level <= min_level)
            {
                return ca;
            }

            // Without a domain, the cells may only be refined where the input has cells.
            lca_t coverage;
            for (std::size_t l = min_level; l <= max_level; ++l)
            {
                coverage = lca_t(union_(coverage, self(ca[l]).on(max_level)));
            }
            const auto round_and_clip = [&](const lca_t& x, std::size_t l)
            {
                const lca_t rounded = (l > min_level) ? lca_t(self(x).on(l - 1).on(l)) : x;
                return domain.empty() ? lca_t(intersection(rounded, self(coverage).on(l))) : lca_t(intersection(rounded, domain[l]));
            };

            std::array<lca_t, default_config::max_level + 1> F;
            const auto graded = [&](std::size_t l)
            {
                return (l < max_level) ? lca_t(difference(F[l], self(F[l + 1]).on(l))) : F[l];
            };
            // The cells a wide stencil needs along the physical boundary (see make_graduation).
            const auto case1 = [&](std::size_t l, const lca_t& g)
            {
                const int n_contig = std::max(radius, 2 * (radius - 2));
                lca_t cells;
                for_each_cartesian_direction<dim>(
                    [&](const auto d, const auto& t)
                    {
                        if (is_periodic[d])
                        {
                            return;
                        }
                        const lca_t on_face(difference(g, translate(domain[l], -t)));
                        for (int i = 2; i <= n_contig; i += 2)
                        {
                            cells = lca_t(union_(cells, intersection(translate(on_face, -i * t), domain[l])));
                        }
                    });
                return cells;
            };
            const auto case2 = [&](std::size_t l, const lca_t& g)
            {
                lca_t cells;
                const lca_t g_coarse(self(g).on(l - 1));
                for_each_cartesian_direction<dim>(
                    [&](const auto d, const auto& t)
                    {
                        if (is_periodic[d])
                        {
                            return;
                        }
                        const lca_t face(difference(domain[l - 1], translate(domain[l - 1], -t)));
                        for (int i = 1; i < radius; ++i)
                        {
                            const lca_t coarse(intersection(face, translate(g_coarse, i * t)));
                            cells = lca_t(union_(cells, self(coarse).on(l)));
                        }
                    });
                return cells;
            };
            const auto extend_boundary = [&](std::size_t l)
            {
                if (radius <= 1 || domain.empty())
                {
                    return;
                }
                lca_t before;
                do
                {
                    before = F[l];
                    if (radius > 2 && l > 0)
                    {
                        F[l] = round_and_clip(lca_t(union_(F[l], case2(l, graded(l)))), l);
                    }
                    F[l] = round_and_clip(lca_t(union_(F[l], case1(l, graded(l)))), l);
                } while (!(F[l] == before));
            };

            std::array<int, dim> nb_cells_finest_level{};
            for (std::size_t d = 0; d < dim; ++d)
            {
                nb_cells_finest_level[d] = domain.empty()
                                             ? 0
                                             : domain[domain.max_level()].max_indices()[d] - domain[domain.max_level()].min_indices()[d];
            }
            F[max_level] = ca[max_level];
            extend_boundary(max_level);
            for (std::size_t l = max_level; l-- > min_level;)
            {
                const lca_t expanded(expand(self(F[l + 1]).on(l), static_cast<int>(grad_width)));
                lca_t required(union_(ca[l], expanded));
                if (std::any_of(is_periodic.begin(),
                                is_periodic.end(),
                                [](bool p)
                                {
                                    return p;
                                }))
                {
                    for (const auto& d :
                         detail::get_periodic_directions(nb_cells_finest_level, int(domain.max_level()) - int(l), is_periodic))
                    {
                        required = lca_t(union_(required, translate(expanded, d)));
                    }
                }
                F[l] = round_and_clip(required, l);
                extend_boundary(l);
            }

            CellArray<dim> result;
            for (std::size_t l = min_level; l <= max_level; ++l)
            {
                result[l] = graded(l);
            }
            return result;
        }

        // The domain pyramid of a mesh built from `reference`: the domain at every level.
        template <std::size_t dim>
        CellArray<dim> pyramid(const LevelCellArray<dim>& reference, std::size_t top_level)
        {
            CellArray<dim> domain;
            for (std::size_t l = 0; l <= std::max(top_level, reference.level()); ++l)
            {
                domain[l] = LevelCellArray<dim>(self(reference).on(l));
            }
            return domain;
        }

        // Random adapted cells, as an adaptation hands them to make_graduation: the domain at
        // min_level, refined around a few random spots and at random isolated cells, without
        // any grading. Sometimes a finest cell is dropped, leaving an incomplete set of
        // siblings, or finest cells are added outside the domain.
        template <std::size_t dim>
        CellArray<dim> random_cells(const CellArray<dim>& domain, std::size_t min_level, std::size_t max_level, std::mt19937& gen)
        {
            using lca_t = LevelCellArray<dim>;
            std::uniform_real_distribution<double> uniform(0., 1.);
            const auto lo = domain[max_level].min_indices();
            const auto hi = domain[max_level].max_indices();
            std::vector<std::pair<std::array<double, dim>, double>> spots(3);
            for (auto& [center, r] : spots)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    center[d] = lo[d] + uniform(gen) * (hi[d] - lo[d]);
                }
                r = 1. + uniform(gen) * 0.3 * (hi[0] - lo[0]);
            }

            CellArray<dim> ca;
            lca_t current = domain[min_level];
            for (std::size_t l = min_level; l < max_level; ++l)
            {
                LevelCellList<dim> to_refine(l);
                const double scale = static_cast<double>(1 << (max_level - l));
                for_each_cell(current,
                              [&](const auto& cell)
                              {
                                  bool near = false;
                                  for (const auto& [center, r] : spots)
                                  {
                                      double dist2 = 0;
                                      for (std::size_t d = 0; d < dim; ++d)
                                      {
                                          const double x = (cell.indices[d] + 0.5) * scale - center[d];
                                          dist2 += x * x;
                                      }
                                      near = near || dist2 < r * r;
                                  }
                                  if (uniform(gen) < (near ? 0.9 : 0.06))
                                  {
                                      xt::xtensor_fixed<int, xt::xshape<dim - 1>> yz;
                                      for (std::size_t d = 1; d < dim; ++d)
                                      {
                                          yz[d - 1] = cell.indices[d];
                                      }
                                      to_refine[yz].add_point(cell.indices[0]);
                                  }
                              });
                const lca_t refined(to_refine);
                ca[l]   = lca_t(difference(current, refined));
                current = lca_t(intersection(self(refined).on(l + 1), domain[l + 1]));
            }
            ca[max_level] = current;

            if (uniform(gen) < 0.25 && !current.empty())
            {
                std::size_t dropped = std::uniform_int_distribution<std::size_t>(0, current.nb_cells() - 1)(gen);
                LevelCellList<dim> one(max_level);
                for_each_cell(current,
                              [&](const auto& cell)
                              {
                                  if (dropped-- == 0)
                                  {
                                      xt::xtensor_fixed<int, xt::xshape<dim - 1>> yz;
                                      for (std::size_t d = 1; d < dim; ++d)
                                      {
                                          yz[d - 1] = cell.indices[d];
                                      }
                                      one[yz].add_point(cell.indices[0]);
                                  }
                              });
                ca[max_level] = lca_t(difference(current, lca_t(one)));
            }
            if (uniform(gen) < 0.2)
            {
                // The children of a parent beyond the domain along x.
                xt::xtensor_fixed<int, xt::xshape<dim>> min_corner;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    min_corner[d] = 2 * (((d == 0) ? hi[d] + 1 : lo[d]) / 2);
                }
                const lca_t outside(max_level, Box<int, dim>(min_corner, min_corner + 2));
                ca[max_level] = lca_t(union_(ca[max_level], outside));
            }
            return ca;
        }

        template <std::size_t dim>
        void expect_same_as_reference(const LevelCellArray<dim>& domain_reference, const std::array<bool, dim>& is_periodic, unsigned seed)
        {
            struct NoMesh
            {
            };

            const std::vector<MPI_Subdomain<NoMesh>> no_neighbours;
            std::mt19937 gen(seed);
            const std::array<std::pair<std::size_t, int>, 7> width_and_radius{
                {{1, 1}, {1, 2}, {2, 2}, {0, 2}, {1, 3}, {2, 4}, {1, 5}}
            };
            for (int trial = 0; trial < 4; ++trial)
            {
                const std::size_t min_level = 1 + static_cast<std::size_t>(trial % 2);
                const std::size_t max_level = min_level + 2 + static_cast<std::size_t>(trial / 2);
                const auto domain           = pyramid(domain_reference, max_level);
                const auto cells            = random_cells(domain, min_level, max_level, gen);
                for (const auto& [width, radius] : width_and_radius)
                {
                    auto graded = cells;
                    make_graduation(graded, domain, no_neighbours, is_periodic, width, radius);
                    EXPECT_TRUE(graded == reference_graduation(cells, domain, is_periodic, width, radius))
                        << "dim " << dim << ", seed " << seed << ", trial " << trial << ", width " << width << ", radius " << radius;
                }
                // Without a domain, the graduation stays inside the input cells.
                auto graded = cells;
                make_graduation(graded, std::size_t{1});
                EXPECT_TRUE(graded == reference_graduation(cells, CellArray<dim>{}, std::array<bool, dim>{}, 1, 1))
                    << "dim " << dim << ", seed " << seed << ", trial " << trial << ", no domain";
            }
        }

        template <std::size_t dim>
        LevelCellArray<dim> box_lca(std::size_t level, const std::array<int, dim>& lo, const std::array<int, dim>& hi)
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> min_corner, max_corner;
            for (std::size_t d = 0; d < dim; ++d)
            {
                min_corner[d] = lo[d];
                max_corner[d] = hi[d];
            }
            return LevelCellArray<dim>(level, Box<int, dim>(min_corner, max_corner));
        }
    }

    // The graded cells match the reference definition on random adapted meshes, in box
    // domains (periodic or not) and in domains with a hole, for several widths and radii.
    TEST(graduation, same_cells_as_reference_1d)
    {
        const auto box  = box_lca<1>(4, {0}, {16});
        const auto hole = LevelCellArray<1>(difference(box_lca<1>(5, {0}, {23}), box_lca<1>(5, {9}, {13})));
        for (unsigned seed = 0; seed < 4; ++seed)
        {
            expect_same_as_reference<1>(box, {false}, seed);
            expect_same_as_reference<1>(box, {true}, seed);
            expect_same_as_reference<1>(hole, {false}, seed);
        }
    }

    TEST(graduation, same_cells_as_reference_2d)
    {
        const auto box  = box_lca<2>(3, {0, 0}, {8, 6});
        const auto hole = LevelCellArray<2>(difference(box_lca<2>(4, {0, 0}, {24, 20}), box_lca<2>(4, {9, 5}, {15, 13})));
        for (unsigned seed = 0; seed < 3; ++seed)
        {
            expect_same_as_reference<2>(box, {false, false}, seed);
            expect_same_as_reference<2>(box, {true, false}, seed);
            expect_same_as_reference<2>(box, {true, true}, seed);
            expect_same_as_reference<2>(hole, {false, false}, seed);
        }
    }

    TEST(graduation, same_cells_as_reference_3d)
    {
        const auto box  = box_lca<3>(2, {0, 0, 0}, {4, 3, 4});
        const auto hole = LevelCellArray<3>(difference(box_lca<3>(3, {0, 0, 0}, {10, 8, 8}), box_lca<3>(3, {3, 2, 3}, {7, 5, 6})));
        for (unsigned seed = 0; seed < 2; ++seed)
        {
            expect_same_as_reference<3>(box, {false, false, false}, seed);
            expect_same_as_reference<3>(box, {false, true, true}, seed);
            expect_same_as_reference<3>(hole, {false, false, false}, seed);
        }
    }
}
