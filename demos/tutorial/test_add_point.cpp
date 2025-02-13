// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

int main()
{
/*
	constexpr std::size_t dim = 1; // cppcheck-suppress unreadVariable

	samurai::CellList<dim> cl;

	cl[0][{}].add_point(0);
	cl[0][{}].add_point(1);
	cl[0][{}].add_point(4);
	cl[0][{}].add_point(5);
	cl[0][{}].add_point(8);
	cl[0][{}].add_point(9);
	cl[0][{}].add_point(15);
	cl[0][{}].add_point(16);

	samurai::CellArray<dim> ca{cl};

	std::cout << "ca = " << std::endl << ca << std::endl;

	ca[0].add_point({3});
	std::cout << "ca = " << std::endl << ca << std::endl;
	ca[0].add_point(2);
	std::cout << "ca = " << std::endl << ca << std::endl;
	ca[0].add_point({20});
	std::cout << "ca = " << std::endl << ca << std::endl;

	for (int i=0;i<21;++i) { ca[0].add_point({i}); }
	std::cout << "ca = " << std::endl << ca << std::endl;

	ca[0].remove_point({10});
	std::cout << "ca = " << std::endl << ca << std::endl;

	ca[0].remove_point({0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	ca[0].remove_point({20});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	for (int i=0;i<21;++i) { ca[0].remove_point({i}); }
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	for (int i=0;i<21;++i) { ca[0].add_point({i}); }
	std::cout << "ca = " << std::endl << ca << std::endl;
*/
	constexpr size_t dim = 3;
	
	samurai::CellArray<dim> ca;
	
	ca[0].add_point({0, 0 , 0});
	ca[0].add_point({0, 1 , 0});
	ca[0].add_point({0, 4 , 0});
	ca[0].add_point({0, 5 , 0});
	ca[0].add_point({0, 8 , 0});
	ca[0].add_point({0, 9 , 0});
	ca[0].add_point({0, 15, 0});
	ca[0].add_point({0, 16, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	ca[0].add_point({0, 3, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	ca[0].add_point({0, 2, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	ca[0].add_point({0, 20, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	for (int i=0;i<21;++i) { ca[0].add_point({0, i, 0}); }
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	ca[0].remove_point({0, 10, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	ca[0].remove_point({0, 0, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	ca[0].remove_point({0, 20, 0});
	std::cout << "ca = " << std::endl << ca << std::endl;
	
	for (int i=0;i<21;++i) { ca[0].remove_point({0, i, 0}); }
	std::cout << "ca = " << std::endl << ca << std::endl;

	for (int i=0;i<21;++i) { ca[0].add_point({0, i, 0}); }
	std::cout << "ca = " << std::endl << ca << std::endl;

	return 0;
}
