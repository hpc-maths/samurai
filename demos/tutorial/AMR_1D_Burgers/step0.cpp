#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <xtensor/xmath.hpp>

using std::pow;
using xt::pow;
using std::exp;
using xt::exp;

int main()
{
    std::size_t max_level = 8;
    std::size_t min_level = 2;
    
    constexpr std::size_t dim = 1;
    samurai::Box<double, dim> box({-3}, {3});
    samurai::CellArray<dim> mesh;

    mesh[min_level] = {min_level, box};

    double sigma = .05;

    auto my_function = [sigma] (auto x)
    {
        return -2*x/(sigma*sigma)*exp(-pow(x, 2.)/pow(sigma, 2.));
    };

    for(std::size_t nite = 0; nite < max_level - min_level; ++nite)
    {
        samurai::CellList<dim> cl;
        samurai::for_each_cell(mesh, [&](auto cell)
        {
            if (cell.level == min_level + nite)
            {
                double x = cell.center(0);

    
                if (cell.level < max_level and std::abs(my_function(x))*cell.length > 0.01)
                {
                    cl[cell.level + 1][{}].add_interval({2*cell.indices[0], 2*cell.indices[0] + 2});
                } 
                else
                {
                    cl[cell.level][{}].add_point(cell.indices[0]);
                }
            }
            else
            {
                cl[cell.level][{}].add_point(cell.indices[0]);
            }
            
        });
        mesh = cl;
    }

    auto level_field = samurai::make_field<std::size_t, 1>("level", mesh);
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& i, auto)
    {
        level_field(level, i) = level;
        double dx = 1./(1<<level);
        auto coords = (xt::arange(i.start, i.end) + .5)*dx;
        u(level, i) = my_function(coords);
    });

    samurai::save("Step0", mesh, level_field, u);

    return 0;
}
