#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>


template <unsigned int dim>
auto cell_list_with_n_intervals(int64_t size){
        samurai::CellList<dim> cl ;
        for (int64_t i = 0 ; i < size; i++){
                int index = static_cast<int>(i) ;
                cl[0][{}].add_interval({2*index, 2*index+1});
        }
        return cl ;
}

template <unsigned int dim>
auto cell_array_with_n_intervals(int64_t size){
        auto cl = cell_list_with_n_intervals<dim>(size) ;
        samurai::CellArray<dim> ca(cl) ;
        return ca;
}


template <unsigned int dim>
void FIND_find_begin(benchmark::State& state){
	auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
	xt::xtensor_fixed<int, xt::xshape<dim>> coord = {0} ; 
	for(auto _ : state ){
		auto index = find(ca[0], coord) ; 
		benchmark::DoNotOptimize(index) ; 
	}
}

template <unsigned int dim>
void FIND_find_end(benchmark::State& state){
        auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
        xt::xtensor_fixed<int, xt::xshape<dim>> coord = {2*state.range(0)} ;
        for(auto _ : state ){
                auto index = find(ca[0], coord) ;
                benchmark::DoNotOptimize(index) ;
        }
}

template <unsigned int dim>
void FIND_find_impl_begin(benchmark::State& state){
        auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
	auto lca = ca[0] ; 
        xt::xtensor_fixed<int, xt::xshape<dim>> coord = {0} ;
	auto size = lca[dim-1].size() ; 
	auto integral = std::integral_constant<std::size_t, dim-1>{} ; 
        for(auto _ : state ){
                auto index = samurai::detail::find_impl(lca, 0, size, coord, integral) ;
                benchmark::DoNotOptimize(index) ;
        }
}

template <unsigned int dim>
void FIND_find_impl_end(benchmark::State& state){
        auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
        auto lca = ca[0] ;
        xt::xtensor_fixed<int, xt::xshape<dim>> coord = {2*state.range(0)} ;
        auto size = lca[dim-1].size() ;
        auto integral = std::integral_constant<std::size_t, dim-1>{} ;
        for(auto _ : state ){
                auto index = samurai::detail::find_impl(lca, 0, size, coord, integral) ;
                benchmark::DoNotOptimize(index) ;
        }
}

template <unsigned int dim>
void FIND_interval_search_begin(benchmark::State& state){
	using  TInterval = samurai::default_config::interval_t;
	using lca_t = const samurai::LevelCellArray<dim, TInterval> ; 
	using diff_t = typename lca_t::const_iterator::difference_type ; 

        auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
        auto lca = ca[0] ;
        xt::xtensor_fixed<int, xt::xshape<dim>> coord = {0} ;
        auto size = lca[dim-1].size() ;
        auto integral = std::integral_constant<std::size_t, dim-1>{} ;
	auto begin = lca[0].cbegin() + static_cast<diff_t>(0) ; 
	auto end = lca[0].cend() +static_cast<diff_t>(size); 

        for(auto _ : state ){
		auto index = samurai::detail::interval_search(begin, end, coord[0]) ; 
                benchmark::DoNotOptimize(index) ;
        }
}


template <unsigned int dim>
void FIND_interval_search_end(benchmark::State& state){
        using  TInterval = samurai::default_config::interval_t;
        using lca_t = const samurai::LevelCellArray<dim, TInterval> ;
        using diff_t = typename lca_t::const_iterator::difference_type ;

        auto ca = cell_array_with_n_intervals<dim>(state.range(0)) ;
        auto lca = ca[0] ;
        xt::xtensor_fixed<int, xt::xshape<dim>> coord = {2*state.range(0)} ;
        auto size = lca[dim-1].size() ;
        auto integral = std::integral_constant<std::size_t, dim-1>{} ;
        auto begin = lca[0].cbegin() + static_cast<diff_t>(0) ;
        auto end = lca[0].cend() +static_cast<diff_t>(size);

        for(auto _ : state ){
                auto index = samurai::detail::interval_search(begin, end, coord[0]) ;
                benchmark::DoNotOptimize(index) ;
        }
}



BENCHMARK_TEMPLATE(FIND_find_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_begin,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_begin,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_end,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(FIND_find_impl_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_impl_begin,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_impl_begin,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(FIND_find_impl_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_impl_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_find_impl_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


BENCHMARK_TEMPLATE(FIND_interval_search_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_interval_search_begin,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_interval_search_begin,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(FIND_interval_search_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_interval_search_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(FIND_interval_search_end,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);



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

        samurai::for_each_interval(
            ca,
            [&](std::size_t level, const auto& interval, const auto& index)
            {
                auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
                for (int i = interval.start, ic = 0; i < interval.end; ++i, ++ic)
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
