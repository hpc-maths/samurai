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
	inline void add_point(const size_t level, const int x, const yz_point_t<dim>& yz_point, std::vector< SimpleCell<dim> >& cells)
	{
		xyz_point_t<dim> xyz_point;
		for (size_t i=1;i<dim;++i) { xyz_point[i] = yz_point[i-1]; }
		
		xyz_point[0] = x;
		cells.push_back(SimpleCell<dim>{level, xyz_point});
	}
	
	template<size_t dim>
	inline void remove_point(const size_t level, const int x, const yz_point_t<dim>& yz_point, std::vector< SimpleCell<dim> >& cells)
	{
		using iterator = std::vector< SimpleCell<dim> >::iterator;
		iterator it = std::remove_if(cells.begin(), cells.end(), [level, x, &yz_point](const SimpleCell<dim>& cell)
		{
			const xyz_point_t<dim>& xyz_point = cell.point;
			
			bool isSameYZ = true;
			for (size_t i=1;i<dim;++i) { isSameYZ = (isSameYZ and xyz_point[i] == yz_point[i-1]); }
			return (cell.level == level) and (x == xyz_point[0]) and isSameYZ;
		});
		cells.erase(it, cells.end());
	}

	TEST(add_point, test_1d)
	{
		constexpr size_t dim = 1;

		samurai::CellArray<dim> ca;
		
		std::vector< SimpleCell<dim> > cells;

		ca[0].add_point( 0, {}); add_point(0,  0, {}, cells); 
		ca[0].add_point( 1, {}); add_point(0,  1, {}, cells);
		ca[0].add_point( 4, {}); add_point(0,  4, {}, cells);
		ca[0].add_point( 5, {}); add_point(0,  5, {}, cells);
		ca[0].add_point( 8, {}); add_point(0,  8, {}, cells);
		ca[0].add_point( 9, {}); add_point(0,  9, {}, cells);
		ca[0].add_point(15, {}); add_point(0, 15, {}, cells);
		ca[0].add_point(16, {}); add_point(0, 16, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point(3, {}); add_point(0, 3, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point(2, {}); add_point(0, 2, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].add_point(20, {}); add_point(0, 20, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].add_point(i, {}); add_point(0, i, {}, cells);}
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point(10, {}); remove_point(0, 10, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point(0, {}); remove_point(0, 0, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		ca[0].remove_point(20, {}); remove_point(0, 20, {}, cells);
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].remove_point(i, {}); remove_point(0, i, {}, cells); }
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
		
		for (int i=0;i<21;++i) { ca[0].add_point(i, {}); add_point(0, i, {}, cells); }
		
		{
			samurai::CellArray<dim> ca_ref = createCellArray(cells);
			EXPECT_EQ(ca_ref, ca);
		}
	}
	
	TEST(add_point, torture_test_add_then_remove_3d)
	{
		constexpr size_t dim = 3;

		samurai::CellArray<dim> ca;
		
		std::vector< SimpleCell<dim> > cells;
		
		const int x_min = 0;
		const int x_max = 16;
		const int y_min = 0;
		const int y_max = 16;
		const int z_min = 0;
		const int z_max = 16;
		
		const int area = (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
		
		cells.reserve(100*area);
		
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			ca[0].add_point( x, {y, z});
			add_point(0, x, {y, z}, cells);
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
		for (size_t i=0;i<100*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			ca[0].remove_point(x, {y, z});
			remove_point(0, x, {y, z}, cells);
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}

	TEST(add_point, torture_test_add_or_remove_3d)
	{
		constexpr size_t dim = 3;

		samurai::CellArray<dim> ca;
		
		std::vector< SimpleCell<dim> > cells;
		
		const int x_min = 0;
		const int x_max = 16;
		const int y_min = 0;
		const int y_max = 16;
		const int z_min = 0;
		const int z_max = 16;
		
		const int area = (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
		
		cells.reserve(200*area);
		
		for (size_t i=0;i<200*area;++i)
		{
			const int x = (x_max - x_min)*double(std::rand()) / double(RAND_MAX) + x_min;
			const int y = (y_max - y_min)*double(std::rand()) / double(RAND_MAX) + y_min;
			const int z = (z_max - z_min)*double(std::rand()) / double(RAND_MAX) + z_min;
			
			if (double(std::rand()) / double(RAND_MAX) > 0.5)
			{
				ca[0].add_point( x, {y, z}); add_point(0, x, {y, z}, cells);
			}
			else
			{
				ca[0].remove_point( x, {y, z}); remove_point(0, x, {y, z}, cells);
			}
			
			{
				samurai::CellArray<dim> ca_ref = createCellArray(cells);
				EXPECT_EQ(ca_ref, ca);
			}
		}
	}

}
