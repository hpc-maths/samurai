#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

#include <vector>
#include <array>

namespace samurai
{
	template<size_t dim> using xyz_point_t = std::array<int, dim>;
	template<size_t dim> using  yz_point_t = std::array<int, dim-1>;
	
	template<size_t dim>
	struct SimpleCell
	{
		size_t           level;
		xyz_point_t<dim> point;
	};
	
	template<size_t dim>
	inline samurai::CellArray<dim> createCellArray(const std::vector< SimpleCell<dim> >& cells)
	{
		samurai::CellList<dim> cl;
		
		for (const SimpleCell<dim>& cell : cells)
		{
			const size_t            level = cell.level;
			const xyz_point_t<dim>& point = cell.point;
			
			const int x = point[0];
			xt::xtensor_fixed<int, xt::xshape<dim-1>> yz; for (size_t i=0;i<dim-1;++i) { yz[i] = point[i+1]; }
			cl[level][yz].add_point(x);
		}
		return samurai::CellArray<dim>{cl, true};
	}
	
	template<size_t dim>
	inline void add_interval(const size_t level, const std::array<int, 2>& x_interval, const yz_point_t<dim>& yz_point, std::vector< SimpleCell<dim> >& cells)
	{
		xyz_point_t<dim> xyz_point;
		for (size_t i=1;i<dim;++i) { xyz_point[i] = yz_point[i-1]; }
		
		for (int x=x_interval[0];x<x_interval[1];++x)
		{
			xyz_point[0] = x;
			cells.push_back(SimpleCell<dim>{level, xyz_point});
		}
	}
	
	template<size_t dim>
	inline void remove_interval(const size_t level, const std::array<int, 2>& x_interval, const yz_point_t<dim>& yz_point, std::vector< SimpleCell<dim> >& cells)
	{
		using iterator = std::vector< SimpleCell<dim> >::iterator;
		iterator it = std::remove_if(cells.begin(), cells.end(), [level, &x_interval, &yz_point](const SimpleCell<dim>& cell)
		{
			const xyz_point_t<dim>& xyz_point = cell.point;
			
			bool isSameYZ = true;
			for (size_t i=1;i<dim;++i) { isSameYZ = (isSameYZ and xyz_point[i] == yz_point[i-1]); }
			return (cell.level == level) and (x_interval[0] <= xyz_point[0] and xyz_point[0] < x_interval[1]) and isSameYZ;
		});
		cells.erase(it, cells.end());
	}

	TEST(add_interval, test_onera)
	{
		constexpr size_t dim = 2;

		samurai::CellArray<dim> ca;
		std::vector< SimpleCell<dim> > cells;
		
		ca[0].add_interval({1, 3}, {0});    add_interval(0, {1, 3}, {0}, cells); 
		ca[0].add_interval({0, 4}, {0});    add_interval(0, {0, 4}, {0}, cells);
		ca[0].add_interval({0, 1}, {1});    add_interval(0, {0, 1}, {1}, cells);
		ca[0].add_interval({3, 4}, {1});    add_interval(0, {3, 4}, {1}, cells);
		ca[0].add_interval({0, 1}, {2});    add_interval(0, {0, 1}, {2}, cells); 
		ca[0].add_interval({3, 4}, {2});    add_interval(0, {3, 4}, {2}, cells);
		ca[0].add_interval({0, 3}, {3});    add_interval(0, {0, 3}, {3}, cells);
	                                      
		ca[1].add_interval({2, 6}, {2});    add_interval(1, {2, 6}, {2}, cells);
		ca[1].add_interval({2, 6}, {3});    add_interval(1, {2, 6}, {3}, cells);
		ca[1].add_interval({2, 4}, {4});    add_interval(1, {2, 4}, {4}, cells);
		ca[1].add_interval({5, 6}, {4});    add_interval(1, {5, 6}, {4}, cells);
		ca[1].add_interval({2, 6}, {5});    add_interval(1, {2, 6}, {5}, cells);
		ca[1].add_interval({6, 8}, {6});    add_interval(1, {6, 8}, {6}, cells);
		ca[1].add_interval({6, 7}, {7});    add_interval(1, {6, 7}, {7}, cells);
	                                      
		ca[2].add_interval({8, 10}, {8});   add_interval(2, {8, 10}, {8}, cells);
		ca[2].add_interval({8, 10}, {9});   add_interval(2, {8, 10}, {9}, cells);
		ca[2].add_interval({14, 16}, {14}); add_interval(2, {14, 16}, {14}, cells);
		ca[2].add_interval({14, 16}, {15}); add_interval(2, {14, 16}, {15}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_interval({1, 3}, {0}); remove_interval(0, {1, 3}, {0}, cells);
		
		ca[1].add_interval({2, 6}, {0}); add_interval(1, {2,6}, {0}, cells);
		ca[1].add_interval({2, 6}, {1}); add_interval(1, {2,6}, {1}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_interval({0, 4}, {2}); remove_interval(0, {0, 4}, {2}, cells);
		
		ca[2].add_interval({0, 16}, {8});  add_interval(2, {0, 16}, { 8}, cells);
		ca[2].add_interval({0, 16}, {9});  add_interval(2, {0, 16}, { 9}, cells);
		ca[2].add_interval({0, 16}, {10}); add_interval(2, {0, 16}, {10}, cells);
		ca[2].add_interval({0, 16}, {11}); add_interval(2, {0, 16}, {11}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
	}
	
	TEST(add_interval, torture_test_add_then_remove_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		std::vector< SimpleCell<dim> > cells;
		
		const int x_min = 0;
		const int x_max = 128;
		
		const int area = x_max - x_min;
		
		for (size_t i=0;i<500*area;++i)
		{
			int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			if (x1 > x2) { std::swap(x1, x2); }
			
			ca[0].add_interval({x1, x2}, {}); add_interval(0, {x1, x2}, {}, cells);
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
		
		for (size_t i=0;i<500*area;++i)
		{
			int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			if (x1 > x2) { std::swap(x1, x2); }
			ca[0].remove_interval({x1, x2}, {}); remove_interval(0, {x1, x2}, {}, cells);
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
	
	//TEST(add_interval, torture_test_add_then_remove_3d)
	//{
		//constexpr size_t dim = 3;

		//samurai::CellArray<dim> ca;
		//std::vector< SimpleCell<dim> > cells;
		
		//const int x_min = 0;
		//const int x_max = 128;
		
		//const int area = (x_max - x_min)*(x_max - x_min)*(x_max - x_min);
		
		//for (size_t i=0;i<500*area;++i)
		//{
			//int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int y  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int z  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			//if (x1 > x2) { std::swap(x1, x2); }
			
			//ca[0].add_interval({x1, x2}, {y, z}); add_interval(0, {x1, x2}, {y, z}, cells);
			
			//{
				//samurai::CellArray<dim> ca_ref = createCellArray(cells);
				//EXPECT_EQ(ca_ref, ca);
			//}
		//}
		
		//for (size_t i=0;i<500*area;++i)
		//{
			//int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int y  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			//int z  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			//if (x1 > x2) { std::swap(x1, x2); }
			//ca[0].remove_interval({x1, x2}, {y, z}); remove_interval(0, {x1, x2}, {y, z}, cells);
			
			//{
				//samurai::CellArray<dim> ca_ref = createCellArray(cells);
				//EXPECT_EQ(ca_ref, ca);
			//}
		//}
	//}
	
	TEST(add_interval, torture_test_add_or_remove_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		std::vector< SimpleCell<dim> > cells;
		
		const int x_min = 0;
		const int x_max = 128;
		
		const int area = x_max - x_min;
		
		for (size_t i=0;i<1000*area;++i)
		{
			int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			if (x1 > x2) { std::swap(x1, x2); }
			
			if (double(std::rand()) / double(RAND_MAX) > 0.5)
			{
				ca[0].add_interval({x1, x2}, {}); add_interval(0, {x1, x2}, {}, cells);
			}
			else
			{
				ca[0].remove_interval({x1, x2}, {}); remove_interval(0, {x1, x2}, {}, cells);
			}
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
	
	TEST(add_interval, torture_test_add_or_remove_3d)
	{
		constexpr size_t dim = 3;

		samurai::CellArray<dim> ca;
		std::vector< SimpleCell<dim> > cells;
		
		const int x_min = 0;
		const int x_max = 128;
		
		const int area = (x_max - x_min)*(x_max - x_min)*(x_max - x_min);
		
		for (size_t i=0;i<1000*area;++i)
		{
			int x1 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int x2 = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int y  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			int z  = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			
			if (x1 > x2) { std::swap(x1, x2); }
			
			if (double(std::rand()) / double(RAND_MAX) > 0.5)
			{
				ca[0].add_interval({x1, x2}, {y, z}); add_interval(0, {x1, x2}, {y, z}, cells);
			}
			else
			{
				ca[0].remove_interval({x1, x2}, {y, z}); remove_interval(0, {x1, x2}, {y, z}, cells);
			}
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
}
