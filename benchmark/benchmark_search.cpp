#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>
#include <functional>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>

////////////////////////////////////////////////////////////
/// utils

constexpr int DEFAULT_X_INTERVALS = 5; // nombre d'intervalles en X pour les cas 2D/3D

template <unsigned int dim>
auto gen_regular_intervals = [](int max_index, unsigned int level = 0, int x_intervals = DEFAULT_X_INTERVALS)
{
    samurai::CellList<dim> cl;

    int interval_size = 1 << level;       // 2^level
    int spacing       = 1 << (level + 1); // 2^(level+1)

    if constexpr (dim == 1)
    {
        // En 1D on garde le comportement précédent : un intervalle par abscisse.
        for (int x = 0; x < max_index; ++x)
        {
            int start = x * spacing;
            cl[level][{}].add_interval({start, start + interval_size});
        }
    }
    else if constexpr (dim == 2)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                xt::xtensor_fixed<int, xt::xshape<1>> coord{y};
                cl[level][coord].add_interval({start, end});
            }
        }
    }
    else if constexpr (dim == 3)
    {
        int nx = x_intervals;
        for (int x = 0; x < nx; ++x)
        {
            int start = x * spacing;
            int end   = start + interval_size;
            for (int y = 0; y < max_index; ++y)
            {
                for (int z = 0; z < max_index; ++z)
                {
                    xt::xtensor_fixed<int, xt::xshape<2>> coord{y, z};
                    cl[level][coord].add_interval({start, end});
                }
            }
        }
    }

    return cl;
};

template <unsigned int dim>
auto cell_array_with_n_intervals(int max_index)
{
    auto cl = gen_regular_intervals<dim>(max_index, 0, DEFAULT_X_INTERVALS);
    samurai::CellArray<dim> ca(cl);
    return ca;
}

//////////////////////////////////////////////////////////////

// Fonction unifiée pour les benchmarks de recherche
template <unsigned int dim>
void FIND_find_unified(benchmark::State& state, const std::function<xt::xtensor_fixed<int, xt::xshape<dim>>()>& coord_generator)
{
    auto ca = cell_array_with_n_intervals<dim>(state.range(0));

    // Compter le nombre d'intervalles
    std::size_t nb_intervals = 0;
    samurai::for_each_interval(ca,
                               [&](std::size_t level, const auto& interval, const auto& index)
                               {
                                   nb_intervals++;
                               });

    // Compter le nombre d'intervalles dans la direction x
    std::size_t nb_intervals_x = state.range(0);
    auto coord                 = coord_generator();
    for (auto _ : state)
    {
        auto index = find(ca[0], coord);
        benchmark::DoNotOptimize(index);
    }

    state.counters["nb_intervals"]   = nb_intervals;
    state.counters["nb_intervals_x"] = nb_intervals_x;
    state.counters["dimension"]      = dim;

    state.SetItemsProcessed(state.iterations());
}

// Fonctions spécialisées pour chaque politique
template <unsigned int dim>
void FIND_find_start(benchmark::State& state)
{
    auto coord_generator = []()
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> coord;
        coord.fill(0);
        return coord;
    };
    FIND_find_unified<dim>(state, coord_generator);
}

template <unsigned int dim>
void FIND_find_end(benchmark::State& state)
{
    auto coord_generator = [&state]()
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> coord;
        coord.fill(state.range(0) - 1);
        coord[0] = 2 * state.range(0) - 2;
        return coord;
    };
    FIND_find_unified<dim>(state, coord_generator);
}

template <unsigned int dim>
void FIND_find_middle(benchmark::State& state)
{
    auto coord_generator = [&state]()
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> coord;
        // Coordonnées du milieu pour toutes les dimensions sauf x
        for (std::size_t i = 1; i < dim; ++i)
        {
            coord[i] = (state.range(0) - 1) / 2;
        }
        // Coordonnée x au milieu, mais assurée d'être un multiple de 2
        int middle_x = (2 * state.range(0) - 1) / 2; // Milieu de la plage [0, 2*state.range(0) - 1]
        coord[0]     = (middle_x / 2) * 2;           // S'assurer que c'est un multiple de 2
        return coord;
    };
    FIND_find_unified<dim>(state, coord_generator);
}

// Ajusté pour ~10k intervalles max : 1D=10000, 2D=2000, 3D=45
BENCHMARK_TEMPLATE(FIND_find_start, 1)->Args({2})->Args({32})->Args({1000})->Args({10000});
BENCHMARK_TEMPLATE(FIND_find_start, 2)->Args({2})->Args({32})->Args({200})->Args({2000});
BENCHMARK_TEMPLATE(FIND_find_start, 3)->Args({2})->Args({8})->Args({20})->Args({45});
BENCHMARK_TEMPLATE(FIND_find_end, 1)->Args({2})->Args({32})->Args({1000})->Args({10000});
BENCHMARK_TEMPLATE(FIND_find_end, 2)->Args({2})->Args({32})->Args({200})->Args({2000});
BENCHMARK_TEMPLATE(FIND_find_end, 3)->Args({2})->Args({8})->Args({20})->Args({45});
BENCHMARK_TEMPLATE(FIND_find_middle, 1)->Args({2})->Args({32})->Args({1000})->Args({10000});
BENCHMARK_TEMPLATE(FIND_find_middle, 2)->Args({2})->Args({32})->Args({200})->Args({2000});
BENCHMARK_TEMPLATE(FIND_find_middle, 3)->Args({2})->Args({8})->Args({20})->Args({45});

/**
template <std::size_t dim>
auto generate_mesh(int bound, std::size_t start_level, std::size_t max_level)
{
    samurai::Box<int, dim> box({-bound << start_level, -bound << start_level, -bound << start_level},
                               {bound << start_level, bound << start_level, bound << start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    for (std::size_t ite = 0; ite < max_level - start_level; ++ite)
    {
        samurai::CellList<dim> cl;

        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index)
                                   {
                                       auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false},
interval.size()); for (int i = interval.start, ic = 0; i < interval.end; ++i, ++ic)
                                       {
                                           if (choice[ic])
                                           {
                                               samurai::static_nested_loop<dim - 1, 0, 2>(
                                                   [&](auto stencil)
                                                   {
                                                       auto new_index = 2 * index + stencil;
                                                       cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                                   });
                                           }
                                           else
                                           {
                                               cl[level][index].add_point(i);
                                           }
                                       }
                                   });

        ca = {cl, true};
    }

    return ca;
}



template <std::size_t dim_, int bound>
class MyFixture : public ::benchmark::Fixture
{
  public:

    static constexpr std::size_t dim       = dim_;
    static constexpr std::size_t min_level = 1;
    static constexpr std::size_t max_level = 10;

    MyFixture()
    {
        mesh = generate_mesh<dim_>(bound, min_level, max_level);
    }

    void bench(benchmark::State& state)
    {
        std::size_t found = 0;
        for (auto _ : state)
            {
                auto level = std::experimental::randint(min_level, max_level);
//              std::array<int, dim> coord;
        xt::xtensor_fixed<int, xt::xshape<dim>> coord ;
                for (auto& c : coord)
                {
                    c = std::experimental::randint(-bound << level, (bound << level) - 1);
                }
                auto out = samurai::find(mesh[level], coord);
                if (out != -1)
                {
                    found++;
                }
            }
        }
        state.counters["nb cells"] = mesh.nb_cells();
        state.counters["found"]    = static_cast<double>(found) / state.iterations();
    }

    samurai::CellArray<dim_> mesh;
};

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_1D, 1, 1000)

(benchmark::State& state)
{
    bench(state);
}



BENCHMARK_REGISTER_F(MyFixture, Search_1D)->DenseRange(1, 10, 1);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_2D, 2, 10)

(benchmark::State& state)
{
    bench(state);
}

BENCHMARK_REGISTER_F(MyFixture, Search_2D)->DenseRange(1, 10, 1);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_3D, 3, 1)(benchmark::State& state)
{
    bench(state);
}

BENCHMARK_REGISTER_F(MyFixture, Search_3D)->DenseRange(1, 10, 1);

**/
