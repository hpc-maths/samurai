#include <benchmark/benchmark.h>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

/// utils
template <unsigned int dim>
auto unitary_box()
{
    using value_t = samurai::default_config::value_t;
    using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
    point_t point1;
    point_t point2;
    if constexpr (dim == 1)
    {
        point1 = {0};
        point2 = {1};
    }
    if constexpr (dim == 2)
    {
        point1 = {0, 0};
        point2 = {1, 1};
    }
    if constexpr (dim == 3)
    {
        point1 = {0, 0, 0};
        point2 = {1, 1, 1};
    }

    samurai::Box<double, dim> box = samurai::Box<double, dim>(point1, point2);
    return box;
}

/// field creation wrapper
template <unsigned int n_comp, class mesh_t>
auto make_field_wrapper(const std::string& name, mesh_t& mesh)
{
    if constexpr (n_comp == 1)
    {
        return samurai::make_scalar_field<double>(name, mesh);
    }
    else
    {
        return samurai::make_vector_field<double, n_comp>(name, mesh);
    }
}

// Mesure : Allocation d'un champ d'un maillage uniforme de taille de coté n
template <unsigned int dim, unsigned int n_comp>
void FIELD_make_field_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        auto u = make_field_wrapper<n_comp>("u", mesh);
        benchmark::DoNotOptimize(u.array());
    }
}

// Mesure : Remplissage d'un champ 1D de taille de coté n
template <unsigned int dim, unsigned int n_comp>
void FIELD_fill_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        u.fill(1.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 1);
}

// Mesure ; Remplissage d'un champ 1D de taille de coté n ,en utilisant for_each_cell (mesure d'overhead)
template <unsigned int dim, unsigned int n_comp>
void FIELD_for_each_cell_fill_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          u[cell] = 1.0;
                      });
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 1);
}

// Weird : MPI issue ??? wtf ???
template <unsigned int dim>
void FIELD_equal_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<1>(std::string("u"), mesh);
    auto v                        = make_field_wrapper<1>(std::string("v"), mesh);
    u.fill(1.0);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        v = u;
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(1)
                            * static_cast<int64_t>(sizeof(double)) * 2);
}

// Mesure : Ajout d'un scalaire à un champ par broadcasting
template <unsigned int dim, unsigned int n_comp>
void FIELD_add_scalar_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);
    u.fill(1.0);
    auto v = make_field_wrapper<n_comp>("v", mesh);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        v = u + 2.0;
        benchmark::DoNotOptimize(v[0]);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 2);
}

// Mesure : Ajout d'un scalaire à un champ par "for_each_cell"
template <unsigned int dim, unsigned int n_comp>
void FIELD_for_each_cell_add_scalar_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);
    u.fill(1.0);
    auto v = make_field_wrapper<n_comp>("v", mesh);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          v[cell] = u[cell] + 1.0;
                      });
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 2);
}

// Mesure : Somme de deux champs 1D par expression
template <unsigned int dim, unsigned int n_comp>
void FIELD_add_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);
    u.fill(1.0);
    auto v = make_field_wrapper<n_comp>("v", mesh);
    v.fill(1.0);
    auto w = make_field_wrapper<n_comp>("w", mesh);
    w.fill(0.0);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        w = u + v;
        benchmark::DoNotOptimize(w[0]);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 3);
}

// Mesure : Somme de deux champs 1D par "for_each_cell"
template <unsigned int dim, unsigned int n_comp>
void FIELD_for_each_cell_add_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));
    auto u                        = make_field_wrapper<n_comp>("u", mesh);
    u.fill(1.0);
    auto v = make_field_wrapper<n_comp>("v", mesh);
    auto w = make_field_wrapper<n_comp>("w", mesh);
    w.fill(0.0);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          w[cell] = u[cell] + v[cell];
                      });
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 3);
}

// Mesure : Expression complexe avec plusieurs opérations arithmétiques
template <unsigned int dim, unsigned int n_comp>
void FIELD_complex_expression_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));

    // Création des champs
    auto v = make_field_wrapper<n_comp>("v", mesh);
    auto w = make_field_wrapper<n_comp>("w", mesh);
    auto x = make_field_wrapper<n_comp>("x", mesh);
    auto z = make_field_wrapper<n_comp>("z", mesh);
    auto u = make_field_wrapper<n_comp>("u", mesh);

    // Initialisation des champs
    v.fill(1.0);
    w.fill(2.0);
    x.fill(3.0);
    z.fill(4.0); // z non nul
    u.fill(0.0);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        u = 2.0 + v + (w * x) / z;
        benchmark::DoNotOptimize(u[0]);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 5);
}

// Mesure : Expression complexe avec plusieurs opérations arithmétiques (version for_each_cell)
template <unsigned int dim, unsigned int n_comp>
void FIELD_for_each_cell_complex_expression_uniform(benchmark::State& state)
{
    samurai::Box<double, dim> box = unitary_box<dim>();
    using Config                  = samurai::UniformConfig<dim>;
    auto mesh                     = samurai::UniformMesh<Config>(box, state.range(0));

    // Création des champs
    auto v = make_field_wrapper<n_comp>("v", mesh);
    auto w = make_field_wrapper<n_comp>("w", mesh);
    auto x = make_field_wrapper<n_comp>("x", mesh);
    auto z = make_field_wrapper<n_comp>("z", mesh);
    auto u = make_field_wrapper<n_comp>("u", mesh);

    // Initialisation des champs
    v.fill(1.0);
    w.fill(2.0);
    x.fill(3.0);
    z.fill(4.0); // z non nul
    u.fill(0.0);

    // Ajouter les statistiques
    auto total_cells              = mesh.nb_cells();
    state.counters["Dimension"]   = dim;
    state.counters["Mesh_level"]  = state.range(0);
    state.counters["Total_cells"] = total_cells;
    state.counters["Components"]  = n_comp;
    state.counters["ns/cell"] = benchmark::Counter(total_cells, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    for (auto _ : state)
    {
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          u[cell] = 2.0 + v[cell] + (w[cell] * x[cell]) / z[cell];
                      });
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_cells));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(total_cells) * static_cast<int64_t>(n_comp)
                            * static_cast<int64_t>(sizeof(double)) * 5);
}

BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_make_field_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_fill_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_fill_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_fill_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_fill_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_equal_uniform, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_equal_uniform, 2)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_equal_uniform, 3)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_add_scalar_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_scalar_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_add_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_add_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_add_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_add_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_add_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_add_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_add_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_complex_expression_uniform, 3, 4)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 1, 1)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 2, 1)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 3, 1)->Arg(5);

BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 1, 4)->Arg(16);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 2, 4)->Arg(8);
BENCHMARK_TEMPLATE(FIELD_for_each_cell_complex_expression_uniform, 3, 4)->Arg(5);
