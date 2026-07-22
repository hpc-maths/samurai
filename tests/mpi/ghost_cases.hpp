// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Catalog of the meshes and domain decompositions exercised by the parallel
// ghost-update robustness suite (tests/mpi/test_ghost_update_parallel.cpp).
//
// Everything a case is made of - the physical domain shapes, the mesh
// geometries, and the (tangled) partitions - lives here, free of GoogleTest, so
// that BOTH the test and the demos/mpi/ghost_cases visualisation tool build the
// exact same cases from a single source of truth. The test adds the oracles and
// the GoogleTest wiring on top; the demo adds the HDF5/XDMF dump on top.
//
// See the header comment of test_ghost_update_parallel.cpp for the rationale of
// each geometry and decomposition.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

#include "lambda_strategy.hpp"

namespace samurai::ghost_cases
{
    namespace lb = samurai::load_balancing;

    using samurai_test::LambdaStrategy;

    // Coarsest level per dimension: 2D can afford a fine coarsest level (needed
    // by the analytic oracle's boundary margin); 3D is kept coarser for cost.
    template <std::size_t Dim>
    struct config
    {
        static constexpr std::size_t base = (Dim == 2) ? 5 : 4;
        using Mesh                        = samurai::MRMesh<samurai::mesh_config<Dim>>;
        using mesh_id_t                   = typename Mesh::mesh_id_t;
    };

    // ---- affine oracle field ---------------------------------------------

    template <std::size_t Dim, class Cell>
    double affine_at_center(const Cell& cell)
    {
        constexpr double coef[3] = {3., 5., 7.};
        double v                 = 2.;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            v += coef[d] * cell.center(d);
        }
        return v;
    }

    template <std::size_t Dim, class Coords>
    double affine_at_coords(const Coords& c)
    {
        constexpr double coef[3] = {3., 5., 7.};
        double v                 = 2.;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            v += coef[d] * c[d];
        }
        return v;
    }

    // stencil size -> ghost width. ghost_width = max(ceil(size/2), prediction
    // radius = 1); since ceil(size/2) >= 1 for size >= 1 this is just the radius.
    inline int ghost_width_of(int stencil_size)
    {
        return stencil_size / 2 + (stencil_size % 2);
    }

    // ---- mesh geometries -------------------------------------------------

    // Physical domain [lo, hi] per dimension. Non-unit boxes (different extent
    // per dimension, shifted or negative origins) exercise anisotropic cell
    // counts, non-zero / negative cell indices and per-dimension periodic shifts.
    template <std::size_t Dim>
    using DomainCorner = xt::xtensor_fixed<double, xt::xshape<Dim>>;

    // Two-level mesh (base / base+1): refine the coarse cells selected by
    // `refine_here`. A two-level mesh is graduated by construction (neighbouring
    // levels never differ by more than one), so any block-like predicate is
    // valid. Threads the domain box, the stencil size and the periodicity.
    template <std::size_t Dim, class Pred>
    typename config<Dim>::Mesh
    make_two_level_mesh(const DomainCorner<Dim>& lo, const DomainCorner<Dim>& hi, int stencil_size, bool periodic, Pred&& refine_here)
    {
        using Mesh        = typename config<Dim>::Mesh;
        using mesh_id_t   = typename config<Dim>::mesh_id_t;
        using value_t     = typename Mesh::interval_t::value_t;
        constexpr auto bl = config<Dim>::base;

        const samurai::Box<double, Dim> box(lo, hi);

        auto coarse_cfg = samurai::mesh_config<Dim>()
                              .min_level(bl)
                              .max_level(bl)
                              .max_stencil_size(stencil_size)
                              .periodic(periodic)
                              .disable_minimal_ghost_width();
        auto coarse = samurai::mra::make_mesh(box, coarse_cfg);

        typename Mesh::cl_type cl;
        samurai::for_each_cell(coarse[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                   if (!refine_here(cell))
                                   {
                                       cl[bl][yz].add_point(cell.indices[0]);
                                       return;
                                   }
                                   const auto i = cell.indices[0];
                                   xt::xtensor_fixed<value_t, xt::xshape<Dim - 1>> yz_child;
                                   for (unsigned m = 0; m < (1U << (Dim - 1)); ++m)
                                   {
                                       for (std::size_t d = 0; d + 1 < Dim; ++d)
                                       {
                                           yz_child(d) = 2 * yz(d) + static_cast<value_t>((m >> d) & 1U);
                                       }
                                       cl[bl + 1][yz_child].add_interval({2 * i, 2 * i + 2});
                                   }
                               });

        auto cfg = samurai::mesh_config<Dim>()
                       .min_level(bl)
                       .max_level(bl + 1)
                       .max_stencil_size(stencil_size)
                       .periodic(periodic)
                       .disable_minimal_ghost_width();
        return samurai::mra::make_mesh(cl, cfg);
    }

    template <std::size_t Dim>
    DomainCorner<Dim> filled_corner(double v)
    {
        DomainCorner<Dim> c;
        c.fill(v);
        return c;
    }

    // One refined quadrant/octant reaching the origin corner: a single straight
    // level jump that touches the (possibly periodic) boundary. The predicate is
    // domain-relative (normalised coordinates) so it works on any box.
    template <std::size_t Dim>
    typename config<Dim>::Mesh make_corner_mesh(const DomainCorner<Dim>& lo, const DomainCorner<Dim>& hi, int stencil_size, bool periodic)
    {
        return make_two_level_mesh<Dim>(lo,
                                        hi,
                                        stencil_size,
                                        periodic,
                                        [lo, hi](const auto& cell)
                                        {
                                            for (std::size_t d = 0; d < Dim; ++d)
                                            {
                                                const double t = (cell.center(d) - lo[d]) / (hi[d] - lo[d]);
                                                if (t >= 0.5)
                                                {
                                                    return false;
                                                }
                                            }
                                            return true;
                                        });
    }

    // Several disjoint refined blocks spread over the domain (2D geometry, unit
    // cube; uses the first two coordinates only, valid at any Dim).
    template <std::size_t Dim>
    typename config<Dim>::Mesh make_scattered_patches(int stencil_size, bool periodic)
    {
        return make_two_level_mesh<Dim>(filled_corner<Dim>(0.),
                                        filled_corner<Dim>(1.),
                                        stencil_size,
                                        periodic,
                                        [](const auto& cell)
                                        {
                                            const double x = cell.center(0);
                                            const double y = cell.center(1);
                                            auto in        = [&](double x0, double x1, double y0, double y1)
                                            {
                                                return x > x0 && x < x1 && y > y0 && y < y1;
                                            };
                                            return in(0.15, 0.30, 0.15, 0.30) || in(0.62, 0.82, 0.20, 0.35) || in(0.20, 0.35, 0.62, 0.80)
                                                || in(0.62, 0.82, 0.62, 0.82) || in(0.42, 0.58, 0.42, 0.58);
                                        });
    }

    // Genuinely tangled, graded multi-level mesh built by adapting a multi-front
    // field. MRAdapt guarantees a valid graded mesh; we then freeze it and put an
    // affine field on it. This is the "tarabiscoté" geometry: scattered level
    // jumps in every direction.
    template <std::size_t Dim>
    typename config<Dim>::Mesh make_adapted_complex(int stencil_size, bool periodic)
    {
        constexpr auto bl = config<Dim>::base;
        const samurai::Box<double, Dim> box(xt::zeros<double>({Dim}), xt::ones<double>({Dim}));
        auto cfg = samurai::mesh_config<Dim>()
                       .min_level(bl)
                       .max_level(bl + 2)
                       .max_stencil_size(stencil_size)
                       .periodic(periodic)
                       .disable_minimal_ghost_width();
        auto mesh = samurai::mra::make_mesh(box, cfg);

        auto bump = samurai::make_scalar_field<double>("bump", mesh);
        bump.resize();
        // A handful of spherical fronts scattered across the interior; the sharp
        // tanh transitions force local refinement all around them.
        const std::array<std::array<double, 4>, 5> fronts = {
            {
             {0.32, 0.62, 0.45, 0.10},
             {0.68, 0.34, 0.55, 0.08},
             {0.55, 0.70, 0.40, 0.07},
             {0.40, 0.38, 0.60, 0.09},
             {0.70, 0.66, 0.50, 0.06},
             }
        };
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   const auto c = cell.center();
                                   double v     = 0.;
                                   for (const auto& f : fronts)
                                   {
                                       double r2 = 0.;
                                       for (std::size_t d = 0; d < Dim; ++d)
                                       {
                                           r2 += (c[d] - f[d]) * (c[d] - f[d]);
                                       }
                                       v += std::tanh(80. * (std::sqrt(r2) - f[3]));
                                   }
                                   bump[cell] = v;
                               });
        if (!periodic)
        {
            samurai::make_bc<samurai::Dirichlet<1>>(bump, 0.);
        }
        samurai::make_MRAdapt(bump)(samurai::mra_config().epsilon(1e-3));
        return bump.mesh();
    }

    enum class Geometry
    {
        CornerQuadrant,
        ScatteredPatches,
        AdaptedComplex
    };

    inline const char* geom_name(Geometry g)
    {
        switch (g)
        {
            case Geometry::CornerQuadrant:
                return "corner";
            case Geometry::ScatteredPatches:
                return "patches";
            case Geometry::AdaptedComplex:
                return "adapted";
        }
        return "?";
    }

    // Physical domain shape. NonCubicShifted has a different extent per
    // dimension and a shifted origin (dimension 1 starts at a negative
    // coordinate), so cell indices are non-zero and negative there - stressing
    // index arithmetic, per-dimension periodic shifts and neighbour bounding
    // boxes. Anisotropic cell counts fall out of the anisotropic box.
    enum class DomainShape
    {
        UnitCube,
        NonCubicShifted
    };

    inline const char* domain_name(DomainShape s)
    {
        return s == DomainShape::UnitCube ? "unit" : "skewbox";
    }

    template <std::size_t Dim>
    void domain_bounds(DomainShape s, DomainCorner<Dim>& lo, DomainCorner<Dim>& hi)
    {
        if (s == DomainShape::UnitCube)
        {
            lo.fill(0.);
            hi.fill(1.);
            return;
        }
        const double L[3] = {0.5, -0.75, 1.25};
        const double H[3] = {2.5, 0.75, 2.25};
        for (std::size_t d = 0; d < Dim; ++d)
        {
            lo[d] = L[d];
            hi[d] = H[d];
        }
    }

    template <std::size_t Dim>
    typename config<Dim>::Mesh build_mesh(Geometry geom, int stencil_size, bool periodic)
    {
        switch (geom)
        {
            case Geometry::CornerQuadrant:
                return make_corner_mesh<Dim>(filled_corner<Dim>(0.), filled_corner<Dim>(1.), stencil_size, periodic);
            case Geometry::ScatteredPatches:
                return make_scattered_patches<Dim>(stencil_size, periodic);
            case Geometry::AdaptedComplex:
            default:
                return make_adapted_complex<Dim>(stencil_size, periodic);
        }
    }

    // Build the mesh for a (geometry, domain shape). A non-unit domain is only
    // paired with the corner geometry (the matrix guarantees this), built through
    // the box-aware corner builder; every unit-cube case goes through build_mesh.
    template <std::size_t Dim>
    typename config<Dim>::Mesh build_mesh_on_domain(Geometry geom,
                                                    DomainShape shape,
                                                    int stencil_size,
                                                    bool periodic,
                                                    const DomainCorner<Dim>& lo,
                                                    const DomainCorner<Dim>& hi)
    {
        if (shape == DomainShape::UnitCube)
        {
            return build_mesh<Dim>(geom, stencil_size, periodic);
        }
        return make_corner_mesh<Dim>(lo, hi, stencil_size, periodic);
    }

    // ---- decompositions --------------------------------------------------

    enum class Decomp
    {
        None,               // reference: keep the initial (no-load-balancing) partition
        ThinVStrips,        // vertical strips 1 coarse cell wide, ranks cycling
        FineCheckerboard,   // per-fine-cell checkerboard (splits sibling groups)
        CoarseCheckerboard, // blocks of 4 coarse cells in checkerboard
        DiagonalBands,      // thin diagonal bands, ranks cycling
        Hilbert,            // real space-filling-curve rebalance
        RandomHash          // deterministic per-cell hash: maximally shredded subdomains
    };

    inline const char* decomp_name(Decomp d)
    {
        switch (d)
        {
            case Decomp::None:
                return "none";
            case Decomp::ThinVStrips:
                return "vstrips";
            case Decomp::FineCheckerboard:
                return "finecheck";
            case Decomp::CoarseCheckerboard:
                return "coarsecheck";
            case Decomp::DiagonalBands:
                return "diagbands";
            case Decomp::Hilbert:
                return "hilbert";
            case Decomp::RandomHash:
                return "randomhash";
        }
        return "?";
    }

    // Non-negative modulo: cell indices can be negative (shifted/negative-origin
    // domains), so a plain % could yield an invalid negative rank.
    inline int rank_mod(long s, int size)
    {
        return static_cast<int>(((s % size) + size) % size);
    }

    // Deterministic per-cell hash -> rank. Mixes level and indices so that
    // adjacent cells land on different ranks: this shreds the domain into
    // per-cell islands, the worst case for neighbour detection and exchange.
    template <std::size_t Dim, class Cell>
    int hashed_rank(const Cell& cell, int size)
    {
        std::uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
        auto mix        = [&](std::uint64_t v)
        {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::uint64_t>(cell.level));
        for (std::size_t d = 0; d < Dim; ++d)
        {
            mix(static_cast<std::uint64_t>(static_cast<long>(cell.indices[d])));
        }
        return static_cast<int>(h % static_cast<std::uint64_t>(size));
    }

    inline bool is_tangled(Decomp d)
    {
        // Hilbert cuts can be locally thick; None is the reference; the rest are
        // always tangled (a subdomain's ghosts reach past its direct neighbour).
        return d != Decomp::Hilbert && d != Decomp::None;
    }

    // Column/row index of a cell brought back to the coarsest level.
    template <std::size_t Dim, class Cell>
    long coarse_index(const Cell& cell, std::size_t d)
    {
        return static_cast<long>(cell.indices[d] >> (cell.level - config<Dim>::base));
    }

    template <std::size_t Dim, class Field>
    void apply_decomposition(Decomp d, Field& u)
    {
        switch (d)
        {
            case Decomp::None:
                break;
            case Decomp::Hilbert:
            {
                auto balancer = lb::make_load_balancer<lb::SFC<lb::Hilbert>>();
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
            case Decomp::ThinVStrips:
            {
                auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                                       LambdaStrategy{[](const auto& cell, int, int size)
                                                                      {
                                                                          return rank_mod(coarse_index<Dim>(cell, 0), size);
                                                                      }});
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
            case Decomp::FineCheckerboard:
            {
                auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                                       LambdaStrategy{[](const auto& cell, int, int size)
                                                                      {
                                                                          long s = 0;
                                                                          for (std::size_t d = 0; d < Dim; ++d)
                                                                          {
                                                                              s += static_cast<long>(cell.indices[d]);
                                                                          }
                                                                          return rank_mod(s, size);
                                                                      }});
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
            case Decomp::CoarseCheckerboard:
            {
                auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                                       LambdaStrategy{[](const auto& cell, int, int size)
                                                                      {
                                                                          long s = 0;
                                                                          for (std::size_t d = 0; d < Dim; ++d)
                                                                          {
                                                                              s += coarse_index<Dim>(cell, d) >> 2;
                                                                          }
                                                                          return rank_mod(s, size);
                                                                      }});
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
            case Decomp::DiagonalBands:
            {
                auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                                       LambdaStrategy{[](const auto& cell, int, int size)
                                                                      {
                                                                          long s = 0;
                                                                          for (std::size_t d = 0; d < Dim; ++d)
                                                                          {
                                                                              s += coarse_index<Dim>(cell, d);
                                                                          }
                                                                          return rank_mod(s, size);
                                                                      }});
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
            case Decomp::RandomHash:
            {
                auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                                       LambdaStrategy{[](const auto& cell, int, int size)
                                                                      {
                                                                          return hashed_rank<Dim>(cell, size);
                                                                      }});
                balancer.load_balance(lb::weight::uniform(), u);
                break;
            }
        }
    }

    // Fill an affine field on its cells and attach the affine Dirichlet BC (only
    // when the mesh is not periodic; a periodic mesh needs no BC).
    template <std::size_t Dim, class Field>
    void fill_affine(Field& u, bool periodic)
    {
        using mesh_id_t = typename config<Dim>::mesh_id_t;
        u.fill(0.);
        samurai::for_each_cell(u.mesh()[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   u[cell] = affine_at_center<Dim>(cell);
                               });
        if (!periodic)
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                    [](const auto&, const auto&, const auto& coords)
                                                    {
                                                        return affine_at_coords<Dim>(coords);
                                                    });
        }
    }

    // ---- case matrices ---------------------------------------------------

    // (A) analytic affine oracle matrix (2D): stencil x geometry x decomposition.
    struct Case
    {
        int stencil_size;
        Geometry geom;
        Decomp decomp;
    };

    inline std::vector<Case> make_cases()
    {
        std::vector<Case> cases;
        for (int s : {1, 2, 3, 4, 5})
        {
            for (Geometry g : {Geometry::CornerQuadrant, Geometry::ScatteredPatches, Geometry::AdaptedComplex})
            {
                for (Decomp d :
                     {Decomp::ThinVStrips, Decomp::FineCheckerboard, Decomp::CoarseCheckerboard, Decomp::DiagonalBands, Decomp::Hilbert})
                {
                    cases.push_back({s, g, d});
                }
            }
        }
        return cases;
    }

    inline std::string case_label(const Case& c)
    {
        return std::string("s") + std::to_string(c.stencil_size) + "_" + geom_name(c.geom) + "_" + decomp_name(c.decomp);
    }

    // (B) decomposition-independence matrix (2D + 3D, periodic/Dirichlet).
    struct ICase
    {
        int stencil_size;
        Geometry geom;
        DomainShape domain;
        bool periodic;
        Decomp decomp;
    };

    // The independence oracle is the one that actually exercises the neighbour
    // exchange (all in-domain ghosts, boundary and level jumps included, 2D/3D,
    // periodic/Dirichlet), so it stresses the hardest configurations:
    //   - the multi-level MRAdapt geometry (level jumps land at subdomain
    //     interfaces once a tangled decomposition cuts through it);
    //   - the RandomHash partition (per-cell islands, corner-only contacts
    //     everywhere - the worst case for neighbour detection);
    //   - a non-cubic, origin-shifted domain (anisotropic cell counts, non-zero
    //     and negative indices, per-dimension periodic shifts).
    inline std::vector<ICase> make_icases()
    {
        std::vector<ICase> cases;
        // ghost width 1 and 3 (the extremes); 1-5 swept by the 2D analytic suite.
        for (Geometry g : {Geometry::CornerQuadrant, Geometry::AdaptedComplex})
        {
            for (int s : {2, 5})
            {
                for (bool periodic : {false, true})
                {
                    for (Decomp d : {Decomp::FineCheckerboard, Decomp::Hilbert, Decomp::RandomHash})
                    {
                        cases.push_back({s, g, DomainShape::UnitCube, periodic, d});
                    }
                }
            }
        }
        // Non-cubic, origin-shifted domain (anisotropic cell counts, non-zero and
        // negative indices, per-dimension periodic shifts), PERIODIC only.
        //
        // Deliberately not the Dirichlet variant: on a non-cubic/shifted domain the
        // affine reconstruction is not exact even sequentially (np=1 shows a ~0.27
        // deviation), so the Dirichlet ghosts become decomposition-dependent for a
        // reason that is NOT a neighbour-exchange bug - it lives in samurai's
        // non-cubic-domain handling and warrants a separate investigation. The
        // periodic case is a clean neighbour-exchange test on such a domain: the
        // exchange must still be decomposition independent, and it is.
        for (int s : {2, 5})
        {
            for (Decomp d : {Decomp::FineCheckerboard, Decomp::RandomHash, Decomp::Hilbert})
            {
                cases.push_back({s, Geometry::CornerQuadrant, DomainShape::NonCubicShifted, /*periodic=*/true, d});
            }
        }
        return cases;
    }

    inline std::string icase_label(const ICase& c)
    {
        return std::string("s") + std::to_string(c.stencil_size) + "_" + geom_name(c.geom) + "_" + domain_name(c.domain) + "_"
             + (c.periodic ? "periodic" : "dirichlet") + "_" + decomp_name(c.decomp);
    }
}
