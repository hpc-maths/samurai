#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

// Observation : Cela prend environ 10ns par intervalle
// si on compare 2 intervalles de taille n, cela prendra environ 2n * 10ns

///////////////////////////////////////////////////////////////////
// Fonctions utilitaires pour la génération d'intervalles
///////////////////////////////////////////////////////////////////

template <unsigned int dim>
auto gen_regular_intervals = [](auto& cl, int index, unsigned int level)
{
    // Calcul des paramètres selon le niveau :
    // Niveau L : taille = 2^L, espacement = 2^(L+1)
    int interval_size = 1 << level;       // 2^level
    int spacing       = 1 << (level + 1); // 2^(level+1)
    int start         = index * spacing;
    int end           = start + interval_size;

    if constexpr (dim == 1)
    {
        cl[level][{}].add_interval({start, end});
    }
    else if constexpr (dim == 2)
    {
        for (int y = 0; y < index; ++y)
        {
            xt::xtensor_fixed<int, xt::xshape<1>> coord{y};
            cl[level][coord].add_interval({start, end});
        }
    }
    else if constexpr (dim == 3)
    {
        for (int y = 0; y < index; ++y)
        {
            for (int z = 0; z < index; ++z)
            {
                xt::xtensor_fixed<int, xt::xshape<2>> coord{y, z};
                cl[level][coord].add_interval({start, end});
            }
        }
    }
};

template <unsigned int dim>
auto gen_offset_intervals = [](auto& cl, int index, unsigned int level)
{
    // Calcul des paramètres selon le niveau
    int interval_size = 1 << level;                      // 2^level
    int spacing       = 1 << (level + 1);                // 2^(level+1)
    int start         = index * spacing + interval_size; // Décalage d'une taille pour être disjoint des "same"
    int end           = start + interval_size;

    if constexpr (dim == 1)
    {
        cl[level][{}].add_interval({start, end});
    }
    else if constexpr (dim == 2)
    {
        for (int y = 0; y < index; ++y)
        {
            xt::xtensor_fixed<int, xt::xshape<1>> coord{y};
            cl[level][coord].add_interval({start, end});
        }
    }
    else if constexpr (dim == 3)
    {
        for (int y = 0; y < index; ++y)
        {
            for (int z = 0; z < index; ++z)
            {
                xt::xtensor_fixed<int, xt::xshape<2>> coord{y, z};
                cl[level][coord].add_interval({start, end});
            }
        }
    }
};

template <unsigned int dim>
auto gen_unique_interval = [](auto& cl, int index, unsigned int level)
{
    // Génère un seul grand intervalle qui grandit avec l'index
    // Pour avoir une bounding box similaire aux autres générateurs,
    // on crée un intervalle qui couvre plusieurs "espacements"

    int spacing = 1 << (level + 1); // 2^(level+1)

    // Créer un grand intervalle qui couvre (index+1) espacements complets
    // Cela donne une taille proportionnelle à l'index, comme les autres générateurs
    int interval_size = (index + 1) * spacing;

    if constexpr (dim == 1)
    {
        if (index == 0)
        {
            cl[level][{}].add_interval({0, interval_size});
        }
    }
    else if constexpr (dim == 2)
    {
        for (int y = 0; y < index; ++y)
        {
            xt::xtensor_fixed<int, xt::xshape<1>> coord{y};
            cl[level][coord].add_interval({0, interval_size});
        }
    }
    else if constexpr (dim == 3)
    {
        for (int y = 0; y < index; ++y)
        {
            for (int z = 0; z < index; ++z)
            {
                xt::xtensor_fixed<int, xt::xshape<2>> coord{y, z};
                cl[level][coord].add_interval({0, interval_size});
            }
        }
    }
};

template <unsigned int dim>
auto create_translation_stencil()
{
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
    if constexpr (dim == 1)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<1>>({1});
    }
    else if constexpr (dim == 2)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
    }
    else if constexpr (dim == 3)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<3>>({1, 1, 1});
    }
    return stencil;
}

///////////////////////////////////////////////////////////////////
// Opérations ensemblistes
///////////////////////////////////////////////////////////////////

auto op_difference = [](const auto& a, const auto& b)
{
    return samurai::difference(a, b);
};

auto op_intersection = [](const auto& a, const auto& b)
{
    return samurai::intersection(a, b);
};

auto op_union = [](const auto& a, const auto& b)
{
    return samurai::union_(a, b);
};

///////////////////////////////////////////////////////////////////
// Fonction de benchmark unifiée
///////////////////////////////////////////////////////////////////

template <unsigned int dim, unsigned int level1, unsigned int level2, typename Gen1, typename Gen2, typename Operation>
void SUBSET_unified_benchmark_mixed_levels(benchmark::State& state, Gen1&& gen1, Gen2&& gen2, Operation&& operation)
{
    samurai::CellList<dim> cl1, cl2;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen1(cl1, index, level1);
        gen2(cl2, index, level2);
    }
    samurai::CellArray<dim> ca1(cl1);
    samurai::CellArray<dim> ca2(cl2);

    // Ajouter les statistiques
    auto total_intervals               = ca1[level1].nb_intervals() + ca2[level2].nb_intervals();
    state.counters["Dimension"]        = dim;
    state.counters["Level1"]           = level1;
    state.counters["Level2"]           = level2;
    state.counters["Input1_intervals"] = ca1[level1].nb_intervals();
    state.counters["Input2_intervals"] = ca2[level2].nb_intervals();
    state.counters["ns/interval"]      = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        auto subset      = operation(ca1[level1], ca2[level2]);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

///////////////////////////////////////////////////////////////////
// Fonction de benchmark unifiée pour les opérations imbriquées
///////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
// Benchmarks pour les opérations ensemblistes
///////////////////////////////////////////////////////////////////

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_diff_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_regular_intervals<dim>, op_difference);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_diff_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_offset_intervals<dim>, op_difference);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_diff_single(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_unique_interval<dim>, op_difference);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_intersect_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_regular_intervals<dim>, op_intersection);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_intersect_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_offset_intervals<dim>, op_intersection);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_intersect_single(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_unique_interval<dim>, op_intersection);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_union_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_regular_intervals<dim>, op_union);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_union_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_offset_intervals<dim>, op_union);
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_set_union_single(benchmark::State& state)
{
    SUBSET_unified_benchmark_mixed_levels<dim, level1, level2>(state, gen_regular_intervals<dim>, gen_unique_interval<dim>, op_union);
}

///////////////////////////////////////////////////////////////////
// Benchmarks pour les opérations géométriques
///////////////////////////////////////////////////////////////////

template <unsigned int dim, unsigned int level>
void SUBSET_translate(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl, index, level);
    }
    samurai::CellArray<dim> ca(cl);

    // Créer le stencil de translation
    auto stencil = create_translation_stencil<dim>();

    // Ajouter les statistiques
    auto total_intervals              = ca[level].nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Level"]           = level;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        auto subset      = samurai::translate(ca[level], stencil);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_translate_and_intersect(benchmark::State& state)
{
    samurai::CellList<dim> cl1, cl2;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl1, index, level1);
        gen_regular_intervals<dim>(cl2, index, level2);
    }
    samurai::CellArray<dim> ca1(cl1);
    samurai::CellArray<dim> ca2(cl2);

    // Créer le stencil de translation
    auto stencil = create_translation_stencil<dim>();

    // Ajouter les statistiques
    auto total_intervals               = ca1[level1].nb_intervals() + ca2[level2].nb_intervals();
    state.counters["Dimension"]        = dim;
    state.counters["Level1"]           = level1;
    state.counters["Level2"]           = level2;
    state.counters["Input1_intervals"] = ca1[level1].nb_intervals();
    state.counters["Input2_intervals"] = ca2[level2].nb_intervals();
    state.counters["ns/interval"]      = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        // intersection(translate(ca1, stencil), ca2)
        auto subset = samurai::intersection(samurai::translate(ca1[level1], stencil), ca2[level2]);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

template <unsigned int dim, unsigned int level1, unsigned int level2>
void SUBSET_translate_and_intersect_and_project(benchmark::State& state)
{
    samurai::CellList<dim> cl1, cl2;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl1, index, level1);
        gen_regular_intervals<dim>(cl2, index, level2);
    }
    samurai::CellArray<dim> ca1(cl1);
    samurai::CellArray<dim> ca2(cl2);

    // Créer le stencil de translation
    auto stencil = create_translation_stencil<dim>();

    // Ajouter les statistiques
    auto total_intervals               = ca1[level1].nb_intervals() + ca2[level2].nb_intervals();
    state.counters["Dimension"]        = dim;
    state.counters["Level1"]           = level1;
    state.counters["Level2"]           = level2;
    state.counters["Input1_intervals"] = ca1[level1].nb_intervals();
    state.counters["Input2_intervals"] = ca2[level2].nb_intervals();
    state.counters["ns/interval"]      = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        // intersection(translate(ca1, stencil), ca2) puis projection sur niveau supérieur
        auto project_level = std::max(level1, level2) + 1;
        auto subset        = samurai::intersection(samurai::translate(ca1[level1], stencil), ca2[level2]).on(project_level);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

template <unsigned int dim, unsigned int level>
void SUBSET_translate_and_project(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl, index, level);
    }
    samurai::CellArray<dim> ca(cl);

    // Créer le stencil de translation
    auto stencil = create_translation_stencil<dim>();

    // Ajouter les statistiques
    auto total_intervals              = ca[level].nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Level"]           = level;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        // Translate puis projection sur niveau supérieur
        auto subset = samurai::translate(ca[level], stencil).on(level + 1);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

template <unsigned int dim, unsigned int level>
void SUBSET_self(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl, index, level);
    }
    samurai::CellArray<dim> ca(cl);

    // Ajouter les statistiques
    auto total_intervals              = ca[level].nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Level"]           = level;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        auto subset      = samurai::self(ca[level]);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

template <unsigned int dim, unsigned int level>
void SUBSET_self_and_project(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_regular_intervals<dim>(cl, index, level);
    }
    samurai::CellArray<dim> ca(cl);

    // Ajouter les statistiques
    auto total_intervals              = ca[level].nb_intervals();
    state.counters["Dimension"]       = dim;
    state.counters["Level"]           = level;
    state.counters["Total_intervals"] = total_intervals;
    state.counters["ns/interval"]     = benchmark::Counter(total_intervals,
                                                       benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto total_cells = 0;
        // Self puis projection sur niveau supérieur
        auto subset = samurai::self(ca[level]).on(level + 1);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

///////////////////////////////////////////////////////////////////
// Enregistrement des benchmarks
///////////////////////////////////////////////////////////////////

// Benchmarks pour les opérations ensemblistes en 1D (même niveau) - ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 1, 0, 0)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 0, 0)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 1, 0, 0)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 1, 0, 0)->Arg(1000);

BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 1, 0, 1)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 0, 1)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 1, 0, 1)->Arg(1000);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 1, 0, 1)->Arg(1000);

// Benchmarks pour les opérations ensemblistes en 2D (même niveau) - ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 2, 0, 0)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 2, 0, 0)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 2, 0, 0)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 2, 0, 0)->Arg(45);

BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 2, 0, 1)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 2, 0, 1)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 2, 0, 1)->Arg(45);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 2, 0, 1)->Arg(45);

// Benchmarks pour les opérations ensemblistes en 3D (même niveau) - ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 3, 0, 0)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 3, 0, 0)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 3, 0, 0)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 3, 0, 0)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 3, 0, 0)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 3, 0, 0)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 3, 0, 0)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 3, 0, 0)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 3, 0, 0)->Arg(15);

BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 3, 0, 1)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 3, 0, 1)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 3, 0, 1)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 3, 0, 1)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 3, 0, 1)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 3, 0, 1)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 3, 0, 1)->Arg(15);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 3, 0, 1)->Arg(15);
// BENCHMARK_TEMPLATE(SUBSET_set_union_single, 3, 0, 1)->Arg(15);

// Benchmarks pour les opérations géométriques (niveau unique)
BENCHMARK_TEMPLATE(SUBSET_translate, 1, 0)->Arg(1000); // ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_translate, 2, 0)->Arg(45);   // ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_translate, 3, 0)->Arg(15);   // ~1000 intervalles

BENCHMARK_TEMPLATE(SUBSET_translate_and_project, 1, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_translate_and_project, 2, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_translate_and_project, 3, 0)->Arg(15);

BENCHMARK_TEMPLATE(SUBSET_self, 1, 0)->Arg(1000); // ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_self, 2, 0)->Arg(45);   // ~1000 intervalles
BENCHMARK_TEMPLATE(SUBSET_self, 3, 0)->Arg(15);   // ~1000 intervalles

BENCHMARK_TEMPLATE(SUBSET_self_and_project, 1, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_self_and_project, 2, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_self_and_project, 3, 0)->Arg(15);

// Benchmarks géométriques avec niveaux mixtes (niveau 0 vs autre niveau)
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 3, 0, 0)->Arg(15);

BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 2, 0, 2)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect, 2, 0, 10)->Arg(45); // this bench is weird : the perf ??? maybe a bug.

BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect_and_project, 1, 0, 0)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect_and_project, 2, 0, 0)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_translate_and_intersect_and_project, 3, 0, 0)->Arg(15);

///////////////////////////////////////////////////////////////////
// Benchmarks avec niveaux mixtes (intéressants pour comparer les niveaux)
///////////////////////////////////////////////////////////////////

// Benchmarks 1D : niveau 0 vs niveau 1
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 0, 1)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 0, 1)->Arg(1000);

// Benchmarks 1D : niveau 0 vs niveau 2
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 0, 2)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 0, 2)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 0, 2)->Arg(1000);

// Benchmarks 1D : niveau 1 vs niveau 2
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 1, 2)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 1, 2)->Arg(1000);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 1, 2)->Arg(1000);

// Benchmarks 2D : niveau 0 vs niveau 1
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 2, 0, 1)->Arg(45);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 2, 0, 1)->Arg(45);

// Benchmarks géométriques avec niveaux mixtes
