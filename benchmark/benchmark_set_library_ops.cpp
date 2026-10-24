// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The operations of bench_set_call_sites with the library machinery: the
// field operators applied to the sets of the call sites, and the library
// functions that contain them. Same meshes and arguments as
// bench_set_call_sites, so that BM_Site_X / BM_SiteOp_X is the share of the
// set algebra in the function that evaluates X.
//
// Args: (max_level, number of balls, stencil radius, geometry), see
// bench::mesh_cases.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/schemes/fv.hpp>

#include "mr_fixtures.hpp"
#include "set_call_sites.hpp"
#include "site_inputs.hpp"

namespace
{
    template <std::size_t dim>
    using mesh_t = bench::mr_mesh_t<dim>;

    template <std::size_t dim>
    using mesh_id_t = typename mesh_t<dim>::mesh_id_t;

    template <std::size_t dim>
    bench::AdaptedCase<dim>& get_case(benchmark::State& state, bool moved = false)
    {
        auto& c = bench::adapted_case<dim>(bench::mesh_case(state), moved);
        bench::report_mesh(state, c.mesh);
        return c;
    }

    // The sets of the mesh of get_case, copied once: the sites read them as
    // bench_set_call_sites does.
    template <std::size_t dim>
    const bench::SiteInputs<dim>& get_inputs(benchmark::State& state)
    {
        static std::map<bench::mesh_key_t, bench::SiteInputs<dim>> cache;
        const auto& mesh = get_case<dim>(state).mesh;
        return bench::cached(cache,
                             bench::mesh_case(state).key(),
                             [&]
                             {
                                 return bench::make_site_inputs<dim>(mesh);
                             });
    }

    // Time accumulated by samurai's timers during the benchmark loop, in ms
    // per iteration, to split a library operation into its phases.
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
} // namespace

namespace sites = bench::sites;

/////////////////////////////////////////////////////////////////////////
// The sets of the call sites with the field operator of the library
/////////////////////////////////////////////////////////////////////////

// BM_SiteOp_X applies to each set of site X the operator the library applies
// there. Defined for the sites that apply one field operator to each set;
// the other sites compare with a BM_Op_*.

template <std::size_t dim>
static void BM_SiteOp_GhostProjection(benchmark::State& state)
{
    const auto& in = get_inputs<dim>(state);
    auto& u        = get_case<dim>(state).u;
    for (auto _ : state)
    {
        sites::ghost_projection<bench::SamuraiStatic>(in,
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
    const auto& in                   = get_inputs<dim>(state);
    auto& u                          = get_case<dim>(state).u;
    constexpr std::size_t pred_order = mesh_t<dim>::config_t::prediction_stencil_radius;
    for (auto _ : state)
    {
        sites::ghost_prediction<bench::SamuraiStatic>(in,
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
    const auto& in = get_inputs<dim>(state);
    auto& u        = get_case<dim>(state).u;
    auto detail    = samurai::make_scalar_field<double>("detail", u.mesh(), 0.);
    for (auto _ : state)
    {
        sites::detail<bench::SamuraiStatic>(in,
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
    const auto& in = get_inputs<dim>(state);
    auto& u        = get_case<dim>(state).u;
    auto& mesh     = u.mesh();
    auto detail    = samurai::make_scalar_field<double>("detail", mesh, 0.);
    auto tag       = samurai::make_scalar_field<std::uint8_t>("tag", mesh, std::uint8_t{0});
    for (auto _ : state)
    {
        sites::tag<bench::SamuraiStatic>(in,
                                         [&](std::size_t level, const auto& set)
                                         {
                                             const double eps_l = bench::mr_epsilon
                                                                / static_cast<double>(1 << (dim * (mesh.max_level() - level - 1)));
                                             set.apply_op(samurai::mr_criteria(detail, tag, eps_l, 1. + dim));
                                         });
        benchmark::DoNotOptimize(tag.array().data());
    }
}

/////////////////////////////////////////////////////////////////////////
// Library operations
/////////////////////////////////////////////////////////////////////////

// BM_Op_* time the library operations that contain the sites: the total each
// BM_Site_* is a share of.

// Contains the ghost projection, prediction and outer ghosts.
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

// One adaptation step, the field of the moved fronts on the mesh adapted to
// the unmoved ones: where the time of a step goes, phase by phase.
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
        auto u    = samurai::make_scalar_field<double>("u",
                                                    mesh,
                                                    bench::balls_indicator<dim>(bench::mesh_case(state).nballs, bench::step_shift(mesh)));
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto adapt = samurai::make_MRAdapt(u);
        state.ResumeTiming();

        adapt(samurai::mra_config().epsilon(bench::mr_epsilon));
        benchmark::DoNotOptimize(u.array().data());
    }

    timers.report(state);
}

// MRMesh(cells, reference mesh), as the adaptation builds the mesh of the
// moved fronts. Contains the corners and the prediction ghosts; its phase
// counters split it further.
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

// update_fields from the mesh of the fronts to the mesh of the moved fronts,
// the total of BM_Site_FieldTransfer.
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

// Upwind convection flux, the total of BM_Site_Interfaces.
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

#define SAMURAI_BENCH_LIBRARY_OPS(DIM)                                       \
    BENCHMARK(BM_Op_MeshConstruction<DIM>)->Apply(bench::mesh_args<DIM>);    \
    BENCHMARK(BM_SiteOp_GhostProjection<DIM>)->Apply(bench::mesh_args<DIM>); \
    BENCHMARK(BM_SiteOp_GhostPrediction<DIM>)->Apply(bench::mesh_args<DIM>); \
    BENCHMARK(BM_Op_UpdateGhost<DIM>)->Apply(bench::mesh_args<DIM>);         \
    BENCHMARK(BM_SiteOp_Detail<DIM>)->Apply(bench::mesh_args<DIM>);          \
    BENCHMARK(BM_SiteOp_Tag<DIM>)->Apply(bench::mesh_args<DIM>);             \
    BENCHMARK(BM_Op_FieldTransfer<DIM>)->Apply(bench::mesh_args<DIM>);       \
    BENCHMARK(BM_Op_AdaptStep<DIM>)->Apply(bench::mesh_args<DIM>);           \
    BENCHMARK(BM_Op_UpwindFlux<DIM>)->Apply(bench::mesh_args<DIM>);

SAMURAI_BENCH_LIBRARY_OPS(1)
SAMURAI_BENCH_LIBRARY_OPS(2)
SAMURAI_BENCH_LIBRARY_OPS(3)

BENCHMARK_MAIN();
