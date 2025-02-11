#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>
#include <samurai/amr/mesh.hpp>

template <unsigned int dim>
auto unitary_box(){
	using value_t = samurai::default_config::value_t ; 
	using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
	point_t point1 ; 
	point_t point2 ; 
        if constexpr (dim == 1){
                point1 = {0} ;
                point2 = {1} ;
        }
        if constexpr (dim == 2){
                point1 = {0,0} ;
                point2 = {1,1} ;
        }
        if constexpr (dim == 3){
                point1 = {0,0,0} ;
                point2 = {1,1,1} ;
        }

	samurai::Box<double, dim> box = samurai::Box<double, dim>(point1, point2) ; 
	return box ; 
}





template <unsigned int dim>
void BC_scalar_homogeneous(benchmark::State& state){
        samurai::Box<double, dim> box = unitary_box<dim>() ;
        using Config = samurai::UniformConfig<dim> ;
        auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
        auto u = make_field<double, 1>("u", mesh) ;
        for (auto _ : state){
		samurai::make_bc<samurai::Dirichlet<dim>>(u);
        }
}


template <unsigned int dim>
void BC_vec_homogeneous(benchmark::State& state){
        samurai::Box<double, dim> box = unitary_box<dim>() ;
        using Config = samurai::UniformConfig<dim> ;
        auto mesh = samurai::UniformMesh<Config>(box, state.range(0));
        auto u = make_field<double, 2>("u", mesh) ;
        for (auto _ : state){
                samurai::make_bc<samurai::Dirichlet<dim>>(u);
        }
}



BENCHMARK_TEMPLATE(BC_scalar_homogeneous,1)->DenseRange(1, 16);;
//BENCHMARK_TEMPLATE(BC_scalar_homogeneous,2)->DenseRange(1, 12);; // not enough ghost cells
//BENCHMARK_TEMPLATE(BC_scalar_homogeneous,3)->DenseRange(1, 7);;

BENCHMARK_TEMPLATE(BC_vec_homogeneous,1)->DenseRange(1, 16);;


