// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/hdf5.hpp>

int main()
{
	//constexpr size_t dim = 1;

	//using interval_t = typename samurai::CellArray<dim>::interval_t;
	
	//samurai::CellArray<dim> ca;
	
	//ca[0].add_interval({interval_t(5, 10)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(15, 20)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(25, 30)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(35, 40)});
	//std::cout << ca << std::endl;

	//ca[0].add_interval({interval_t(0, 32)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(32, 35)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(45, 45)});
	//std::cout << ca << std::endl;

	//////////////////////////////////////////////////////////////////////
	
	//constexpr size_t dim = 2;
	//using interval_t = typename samurai::CellArray<dim>::interval_t;
	//samurai::CellArray<dim> ca;
	
	//ca[0].add_interval({interval_t(5, 10), interval_t(5, 10)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(15, 20), interval_t(15, 20)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(25, 30), interval_t(25, 30)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(35, 40), interval_t(35, 40)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(0, 32), interval_t(0, 32)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(32, 35), interval_t(32, 35)});
	//std::cout << ca << std::endl;
	
	//////////////////////////////////////////////////////////////////////

	//constexpr size_t dim = 3;

	//using interval_t = typename samurai::CellArray<dim>::interval_t;
	//samurai::CellArray<dim> ca;
	
	//ca[0].add_interval({interval_t(5, 10), interval_t(5, 10), interval_t(5, 10)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(15, 20), interval_t(15, 20), interval_t(15, 20)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(25, 30), interval_t(25, 30), interval_t(25, 30)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(35, 40), interval_t(35, 40), interval_t(35, 40)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(0, 32), interval_t(0, 32), interval_t(0, 32)});
	//std::cout << ca << std::endl;
	
	//ca[0].add_interval({interval_t(32, 35), interval_t(32, 35), interval_t(32, 35)});
	//std::cout << ca << std::endl;
	
	//////////////////////////////////////////////////////////////////////
		
	// constexpr size_t dim = 1;
	// 
	// samurai::CellArray<dim> ca;
	// 
	// for (size_t i=0;i<10;++i) 
	// { 
	// 	ca[0].add_interval({i, i+1}, {}); 
	// 	std::cout << ca << std::endl;
	// }
	
	//////////////////////////////////////////////////////////////////////
	
	constexpr size_t dim = 2;

	samurai::CellArray<dim> ca;

	ca[0].add_interval({1, 3}, {0}); 
	ca[0].add_interval({0, 4}, {0});
	ca[0].add_interval({0, 1}, {1});
	ca[0].add_interval({3, 4}, {1});
	ca[0].add_interval({0, 1}, {2}); 
	ca[0].add_interval({3, 4}, {2});
	ca[0].add_interval({0, 3}, {3});

	ca[1].add_interval({2, 6}, {2});
	ca[1].add_interval({2, 6}, {3});
	ca[1].add_interval({2, 4}, {4});
	ca[1].add_interval({5, 6}, {4});
	ca[1].add_interval({2, 6}, {5});
	ca[1].add_interval({6, 8}, {6});
	ca[1].add_interval({6, 7}, {7});

	ca[2].add_interval({8, 10}, {8});
	ca[2].add_interval({8, 10}, {9});
	ca[2].add_interval({14, 16}, {14});
	ca[2].add_interval({14, 16}, {15});

	std::cout << ca << std::endl;

	//samurai::Hdf5_CellArray initial_hdf5("", "initial_mesh", {true, true}, ca);
	//initial_hdf5.save();
	
	ca[0].remove_interval({1, 3}, {0});
	ca[1].add_interval({2, 6}, {0});
	ca[1].add_interval({2, 6}, {1});
	
	std::cout << ca << std::endl;
	
	//samurai::Hdf5_CellArray intermediate_hdf5("", "intermediate_mesh", {true, true}, ca);
	//intermediate_hdf5.save();

	ca[0].remove_interval({0, 4}, {2});
	ca[2].add_interval({0, 16}, {8});
	ca[2].add_interval({0, 16}, {9});
	ca[2].add_interval({0, 16}, {10});
	ca[2].add_interval({0, 16}, {11});
	
	std::cout << ca << std::endl;
	
	//samurai::Hdf5_CellArray final_hdf5("", "final_mesh", {true, true}, ca);
	//final_hdf5.save();

	return 0;
}
