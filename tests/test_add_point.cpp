#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

#include <vector>
#include <array>

namespace samurai
{

	template<size_t dim>
	inline samurai::CellArray<dim> createCellArray(const std::vector< std::array<int, dim> >& points)
	{
		samurai::CellList<dim> cl;
		
		for (const std::array<int, dim>& point : points)
		{
			const int x = point[0];
			xt::xtensor_fixed<int, xt::xshape<dim-1>> yz; for (size_t i=0;i<dim-1;++i) { yz[i] = point[i+1]; }
			cl[0][yz].add_point(x);
		}
		return samurai::CellArray<dim>{cl};
	}
	
	template<size_t dim>
	inline void remove_point(std::vector< std::array<int, dim> >& points, const std::array<int, dim> point)
	{
		using const_iterator = std::vector< std::array<int, dim> >::const_iterator;
		const_iterator it = std::remove(points.begin(), points.end(), point);
		points.erase(it, points.end());
	}

	TEST(add_point, test_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		
		std::vector< std::array<int, dim> > points;

		ca[0].add_point({ 0}); points.push_back({ 0}); 
		ca[0].add_point({ 1}); points.push_back({ 1});
		ca[0].add_point({ 4}); points.push_back({ 4});
		ca[0].add_point({ 5}); points.push_back({ 5});
		ca[0].add_point({ 8}); points.push_back({ 8});
		ca[0].add_point({ 9}); points.push_back({ 9});
		ca[0].add_point({15}); points.push_back({15});
		ca[0].add_point({16}); points.push_back({16});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point({3}); points.push_back({3});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point({2}); points.push_back({2});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point({20}); points.push_back({20});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].add_point({i}); points.push_back({i});}
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point({10}); remove_point(points, {10});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point({0}); remove_point(points, {0});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point({20}); remove_point(points, {20});
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].remove_point({i}); remove_point(points, {i}); }
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].add_point({i}); points.push_back({i}); }
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(points);
			EXPECT_EQ(ca_ref, ca);
		}
	}

	TEST(add_point, torture_test_add_then_remove_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		
		std::vector< std::array<int, dim> > points;
		
		const int x_min = 0;
		const int x_max = 128;
		
		const int area = x_max - x_min;
		
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			ca[0].add_point({x});
			points.push_back({x});
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX)  + x_min;
			ca[0].remove_point({x});
			remove_point(points, {x});
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
	
	TEST(add_point, torture_test_add_then_remove_3d)
	{
		constexpr size_t dim = 3;

		samurai::CellArray<dim> ca;
		
		std::vector< std::array<int, dim> > points;
		
		const int x_min = 0;
		const int x_max = 128;
		const int y_min = 0;
		const int y_max = 128;
		const int z_min = 0;
		const int z_max = 128;
		
		const int area = (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
		
		points.reserve(100*area);
		
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			ca[0].add_point( {x, y, z});
			points.push_back({x, y, z});
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			ca[0].remove_point({x, y, z});
			remove_point(points, {x, y, z});
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
	
	TEST(add_point, torture_test_add_or_remove_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		
		std::vector< std::array<int, dim> > points;
		
		const int x_min = 0;
		const int x_max = 128;
		
		const int area = x_max - x_min;
		
		for (size_t i=0;i<200*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX)  + x_min;
			if (double(std::rand()) / double(RAND_MAX) > 0.5)
			{
				ca[0].add_point({x});
				points.push_back({x});
			}
			else
			{
				ca[0].remove_point({x});
				remove_point(points, {x});
			}
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}
	
	TEST(add_point, torture_test_add_or_remove_3d)
	{
		constexpr size_t dim = 3;

		samurai::CellArray<dim> ca;
		
		std::vector< std::array<int, dim> > points;
		
		const int x_min = 0;
		const int x_max = 128;
		const int y_min = 0;
		const int y_max = 128;
		const int z_min = 0;
		const int z_max = 128;
		
		const int area = (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
		
		points.reserve(200*area);
		
		for (size_t i=0;i<200*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			
			if (double(std::rand()) / double(RAND_MAX) > 0.5)
			{
				ca[0].add_point({x,y,z});
				points.push_back({x,y,z});
			}
			else
			{
				ca[0].remove_point({x,y,z});
				remove_point(points, {x,y,z});
			}
			{
				samurai::CellArray<dim> ca_ref = createCellArray(points);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}

}
