// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// Benchmark comparing the OLD vs NEW implementation of the field-update
// set algebra in `update_fields`:
//
//   - OLD: for each field, run its *own* level loops, rebuilding the
//     `intersection` sets once per field (N fields => N set constructions
//     per level).
//   - NEW: run the level loops *once* and, inside each level, apply the
//     operator to ALL fields through a fold expression (N fields => 1 set
//     construction per level).
//
// The workload is built from the first AMR adaptation of the 2D advection
// demo: two adapted meshes (a bump at two slightly different positions, as
// if the solution had been advected by one time step) are captured, then
// the copy / projection / prediction operators are timed from `mesh_old`
// to `mesh_new` with varying numbers of scalar and vector fields.

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include <benchmark/benchmark.h>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/utils.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/samurai.hpp>

namespace
{
    constexpr std::size_t dim = 2;

    using config_t  = samurai::mesh_config<dim>;
    using mesh_t    = samurai::MRMesh<config_t>;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    // ---------------------------------------------------------------------
    // Build an adapted mesh for a "bump" centred at (cx, cy), exactly like
    // the first MR adaptation performed in demos/FiniteVolume/advection_2d.
    // ---------------------------------------------------------------------
    mesh_t make_adapted_mesh(double cx, double cy, double eps)
    {
        samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        auto config = samurai::mesh_config<dim>().min_level(3).max_level(8);
        auto mesh   = samurai::mra::make_mesh(box, config);

        auto u = samurai::make_scalar_field<double>("u", mesh);
        samurai::for_each_cell(
            mesh,
            [&](auto& cell)
            {
                const auto center   = cell.center();
                const double radius = 0.2;
                u[cell] = (((center[0] - cx) * (center[0] - cx) + (center[1] - cy) * (center[1] - cy)) <= radius * radius) ? 1.0 : 0.0;
            });

        auto adapt = samurai::make_MRAdapt(u);
        adapt(samurai::mra_config().epsilon(eps));
        return mesh; // field.mesh() has been swapped to the adapted mesh
    }

    // Shared adapted meshes (built once, lazily): the bump moved from
    // (0.30, 0.30) to (0.35, 0.30) so the two meshes differ across levels.
    struct MeshPair
    {
        mesh_t mesh_old;
        mesh_t mesh_new;
    };

    const MeshPair& meshes()
    {
        static const MeshPair p{make_adapted_mesh(0.30, 0.30, 2e-4),
                                make_adapted_mesh(0.35, 0.30, 2e-4)};
        return p;
    }

    // ---------------------------------------------------------------------
    // Generic helpers to build a tuple of fields living on a mesh.
    // ---------------------------------------------------------------------
    template <class Mesh, std::size_t... Is>
    auto make_scalars_impl(Mesh& m, double init, std::index_sequence<Is...>)
    {
        return std::make_tuple(samurai::make_scalar_field<double>("s" + std::to_string(Is), m, init)...);
    }

    template <std::size_t N>
    auto make_scalars(mesh_t& m, double init)
    {
        return make_scalars_impl(m, init, std::make_index_sequence<N>{});
    }

    auto make_mixed(mesh_t& m, double init)
    {
        return std::make_tuple(samurai::make_scalar_field<double>("ms0", m, init),
                               samurai::make_scalar_field<double>("ms1", m, init),
                               samurai::make_vector_field<double, dim>("mv0", m, init),
                               samurai::make_vector_field<double, dim>("mv1", m, init));
    }

    // Force the timed writes to be observable: sum every value stored in the
    // new fields and escape the result. Applied identically to OLD and NEW, so
    // it does not bias the comparison (it adds the same read cost to both).
    template <class Tuple, std::size_t... Is>
    double tuple_checksum(const Tuple& t, std::index_sequence<Is...>)
    {
        double s = 0.0;
        auto add_one = [&](const auto& f)
        {
            const auto& arr = f.array();
            const auto* p   = arr.data();
            for (std::size_t k = 0, n = arr.size(); k < n; ++k)
            {
                s += static_cast<double>(p[k]);
            }
        };
        (add_one(std::get<Is>(t)), ...);
        return s;
    }

    // =====================================================================
    //                      OLD STYLE (one field at a time)
    // =====================================================================
    template <class Mesh, class Old, class New>
    void old_copy_one(Mesh& om, Mesh& nm, Old& of, New& nf)
    {
        const auto min_l = om.min_level();
        const auto max_l = om.max_level();
        for (std::size_t l = min_l; l <= max_l; ++l)
        {
            auto set = samurai::intersection(om[mesh_id_t::reference][l], nm[mesh_id_t::cells][l]);
            set.apply_op(samurai::copy(nf, of));
        }
    }

    struct OldCopy
    {
        template <class Mesh, class OldT, class NewT, class Seq>
        void operator()(Mesh& om, Mesh& nm, OldT& old, NewT& nw, Seq)
        {
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                (old_copy_one(om, nm, std::get<Is>(old), std::get<Is>(nw)), ...);
            }(Seq{});
        }
    };

    struct Pred
    {
        template <class New, class Old>
        auto operator()(New& nf, const Old& of) const
        {
            constexpr std::size_t pred_order = std::decay_t<New>::mesh_t::config_t::prediction_stencil_radius;
            return samurai::prediction<pred_order, true>(nf, of);
        }
    };

    template <class Mesh, class Old, class New>
    void old_update_one(Mesh& om, Mesh& nm, Old& of, New& nf)
    {
        Pred pred;
        const auto min_l = om.min_level();
        const auto max_l = om.max_level();
        for (std::size_t l = min_l; l <= max_l; ++l)
        {
            auto set = samurai::intersection(om[mesh_id_t::reference][l], nm[mesh_id_t::cells][l]);
            set.apply_op(samurai::copy(nf, of));
        }
        for (std::size_t l = min_l + 1; l <= max_l; ++l)
        {
            auto set_coarsen = samurai::intersection(om[mesh_id_t::cells][l], nm[mesh_id_t::cells][l - 1]).on(l - 1);
            set_coarsen.apply_op(samurai::projection(nf, of));

            auto set_refine = samurai::intersection(nm[mesh_id_t::cells][l], om[mesh_id_t::cells][l - 1]).on(l - 1);
            set_refine.apply_op(pred(nf, of));
        }
    }

    struct OldUpdate
    {
        template <class Mesh, class OldT, class NewT, class Seq>
        void operator()(Mesh& om, Mesh& nm, OldT& old, NewT& nw, Seq)
        {
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                (old_update_one(om, nm, std::get<Is>(old), std::get<Is>(nw)), ...);
            }(Seq{});
        }
    };

    // =====================================================================
    //                  NEW STYLE (single loop, fold over fields)
    // =====================================================================
    // NEW_COPY_FOLD: single loop, one apply per field (fold over fields) —
    // the implementation of `update_fields` on this branch before the
    // variadic-copy operator was introduced.
    struct NewCopyFold
    {
        template <class Mesh, class OldT, class NewT, std::size_t... Is>
        void run(Mesh& om, Mesh& nm, OldT& old, NewT& nw, std::index_sequence<Is...>)
        {
            const auto min_l = om.min_level();
            const auto max_l = om.max_level();
            for (std::size_t l = min_l; l <= max_l; ++l)
            {
                auto set = samurai::intersection(om[mesh_id_t::reference][l], nm[mesh_id_t::cells][l]);
                (set.apply_op(samurai::copy(std::get<Is>(nw), std::get<Is>(old))), ...);
            }
        }

        template <class Mesh, class OldT, class NewT, class Seq>
        void operator()(Mesh& om, Mesh& nm, OldT& old, NewT& nw, Seq)
        {
            run(om, nm, old, nw, Seq{});
        }
    };

    // NEW_COPY_VARIADIC: single loop, ONE apply_op carrying every (dest, src)
    // pair at once (the variadic `copy(tuples)` operator) — a single traversal
    // of the set copies all fields together.
    struct NewCopyVariadic
    {
        template <class Mesh, class OldT, class NewT, class Seq>
        void operator()(Mesh& om, Mesh& nm, OldT& old, NewT& nw, Seq)
        {
            const auto min_l = om.min_level();
            const auto max_l = om.max_level();
            for (std::size_t l = min_l; l <= max_l; ++l)
            {
                auto set = samurai::intersection(om[mesh_id_t::reference][l], nm[mesh_id_t::cells][l]);
                set.apply_op(samurai::copy(nw, old));
            }
        }
    };

    struct NewUpdate
    {
        template <class Mesh, class OldT, class NewT, class Seq>
        void operator()(Mesh& om, Mesh& nm, OldT& old, NewT& nw, Seq)
        {
            const auto min_l = om.min_level();
            const auto max_l = om.max_level();
            
            // Copy phase
            for (std::size_t l = min_l; l <= max_l; ++l)
            {
                auto set = samurai::intersection(om[mesh_id_t::reference][l], nm[mesh_id_t::cells][l]);
                set.apply_op(samurai::copy(nw, old));
            }
            
            // Projection and prediction phases
            constexpr std::size_t pred_order = mesh_t::config_t::prediction_stencil_radius;
            for (std::size_t l = min_l + 1; l <= max_l; ++l)
            {
                auto set_coarsen = samurai::intersection(om[mesh_id_t::cells][l], nm[mesh_id_t::cells][l - 1]).on(l - 1);
                set_coarsen.apply_op(samurai::projection(nw, old));

                auto set_refine = samurai::intersection(nm[mesh_id_t::cells][l], om[mesh_id_t::cells][l - 1]).on(l - 1);
                set_refine.apply_op(samurai::prediction<pred_order, true>(nw, old));
            }
        }
    };

    // =====================================================================
    //                       Benchmark body
    // =====================================================================
    template <class Runner, class BuildOld, class BuildNew, class SeqType>
    void bench_body(benchmark::State& state, const char* style,
                    Runner runner, BuildOld build_old, BuildNew build_new, SeqType)
    {
        const auto& mp = meshes();
        auto om  = mp.mesh_old;
        auto nm  = mp.mesh_new;
        auto old = build_old(om, 1.0);
        auto nw  = build_new(nm, 0.0);
        SeqType seq{};
        const std::size_t nfields = std::tuple_size_v<std::decay_t<decltype(nw)>>;

        for (auto _ : state)
        {
            runner(om, nm, old, nw, seq);
            benchmark::DoNotOptimize(tuple_checksum(nw, seq));
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(nfields));
        state.SetLabel(std::string(style) + " n=" + std::to_string(nfields));
    }

#define SAMURAI_REG3(BenchName, OldRunner, FoldRunner, VarRunner, BuildOld, BuildNew, SeqType)     \
    static void BenchName##_old(benchmark::State& state)                                         \
    {                                                                                            \
        bench_body(state, "OLD", OldRunner{}, BuildOld, BuildNew, SeqType{});                   \
    }                                                                                            \
    static void BenchName##_fold(benchmark::State& state)                                        \
    {                                                                                            \
        bench_body(state, "FOLD", FoldRunner{}, BuildOld, BuildNew, SeqType{});                  \
    }                                                                                            \
    static void BenchName##_var(benchmark::State& state)                                         \
    {                                                                                            \
        bench_body(state, "VAR", VarRunner{}, BuildOld, BuildNew, SeqType{});                    \
    }                                                                                            \
    BENCHMARK(BenchName##_old)->Unit(benchmark::kMillisecond)->UseRealTime();                     \
    BENCHMARK(BenchName##_fold)->Unit(benchmark::kMillisecond)->UseRealTime();                    \
    BENCHMARK(BenchName##_var)->Unit(benchmark::kMillisecond)->UseRealTime()

#define SAMURAI_REG2(BenchName, OldRunner, VarRunner, BuildOld, BuildNew, SeqType)                \
    static void BenchName##_old(benchmark::State& state)                                         \
    {                                                                                            \
        bench_body(state, "OLD", OldRunner{}, BuildOld, BuildNew, SeqType{});                   \
    }                                                                                            \
    static void BenchName##_var(benchmark::State& state)                                         \
    {                                                                                            \
        bench_body(state, "VAR", VarRunner{}, BuildOld, BuildNew, SeqType{});                    \
    }                                                                                            \
    BENCHMARK(BenchName##_old)->Unit(benchmark::kMillisecond)->UseRealTime();                     \
    BENCHMARK(BenchName##_var)->Unit(benchmark::kMillisecond)->UseRealTime()

    // Copy operator only — three strategies.
    SAMURAI_REG3(Copy_1scalar,  OldCopy, NewCopyFold, NewCopyVariadic, make_scalars<1>,  make_scalars<1>,  std::make_index_sequence<1>);
    SAMURAI_REG3(Copy_4scalar,  OldCopy, NewCopyFold, NewCopyVariadic, make_scalars<4>,  make_scalars<4>,  std::make_index_sequence<4>);
    SAMURAI_REG3(Copy_8scalar,  OldCopy, NewCopyFold, NewCopyVariadic, make_scalars<8>,  make_scalars<8>,  std::make_index_sequence<8>);
    SAMURAI_REG3(Copy_16scalar, OldCopy, NewCopyFold, NewCopyVariadic, make_scalars<16>, make_scalars<16>, std::make_index_sequence<16>);
    SAMURAI_REG3(Copy_mixed,   OldCopy, NewCopyFold, NewCopyVariadic, make_mixed,      make_mixed,      std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(make_mixed(std::declval<mesh_t&>(), 0.0))>>>);

    // Full update (variadic copy + per-field projection + per-field prediction).
    SAMURAI_REG2(Update_1scalar,  OldUpdate, NewUpdate, make_scalars<1>,  make_scalars<1>,  std::make_index_sequence<1>);
    SAMURAI_REG2(Update_4scalar,  OldUpdate, NewUpdate, make_scalars<4>,  make_scalars<4>,  std::make_index_sequence<4>);
    SAMURAI_REG2(Update_8scalar,  OldUpdate, NewUpdate, make_scalars<8>,  make_scalars<8>,  std::make_index_sequence<8>);
    SAMURAI_REG2(Update_16scalar, OldUpdate, NewUpdate, make_scalars<16>, make_scalars<16>, std::make_index_sequence<16>);
    SAMURAI_REG2(Update_mixed,   OldUpdate, NewUpdate, make_mixed,      make_mixed,      std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(make_mixed(std::declval<mesh_t&>(), 0.0))>>>);

#undef SAMURAI_REG3
#undef SAMURAI_REG2
}

int main(int argc, char** argv)
{
    samurai::initialize();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    samurai::finalize();
    return 0;
}