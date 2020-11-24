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

    double sigma = .1;

    auto my_function = [sigma] (auto x)
    {
        return exp(-pow(x, 2.)/pow(sigma, 2.));
    };

    auto my_function_der = [sigma, &my_function] (auto x)
    {   
        // Derivative of a gaussian with standard deviation sigma
        return -2*x*my_function(x)/pow(sigma, 2.);
    };

    for(std::size_t nite = 0; nite < max_level - min_level; ++nite)
    {
        samurai::CellList<dim> cl;
        samurai::for_each_cell(mesh, [&](auto cell)
        {
            if (cell.level == min_level + nite)
            {
                double x = cell.center(0);
    
                if (cell.level < max_level and 
                    std::abs(my_function_der(x))*cell.length > 0.01)
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


    // Saving mesh before grading
    {
        auto level_field = samurai::make_field<std::size_t, 1>("level", mesh);
        auto u = samurai::make_field<double, 1>("u", mesh);

        samurai::for_each_interval(mesh, [&](std::size_t level, const auto& i, auto &)
        {
            level_field(level, i) = level;
            double dx = 1./(1<<level);
            auto coords = (xt::arange(i.start, i.end) + .5)*dx;
            u(level, i) = my_function(coords);
        });

        samurai::save("Step0_before", mesh, level_field, u);
    }


    // Grading 
    xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};

    while(true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", mesh);
        tag.fill(false);

        for(std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto set = samurai::intersection(samurai::translate(mesh[level], s), 
                                                     mesh[level_below])
                              .on(level_below);

                    set([&](const auto& i, const auto& )
                    {
                        tag(level_below, i) = true;
                    });
                }
            }
        }

        samurai::CellList<1> cl;
        samurai::for_each_cell(mesh, [&](auto cell)
        {
            auto i = cell.indices[0];
            if (tag[cell])
            {
                cl[cell.level + 1][{}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{}].add_point(i);
            }
        });

        samurai::CellArray<dim> new_mesh = {cl, true};

        if(new_mesh == mesh)
            break;

        std::swap(mesh, new_mesh);
    }

    // Saving graded mesh
    {
        auto level_field = samurai::make_field<std::size_t, 1>("level", mesh);
        auto u = samurai::make_field<double, 1>("u", mesh);

        samurai::for_each_interval(mesh, [&](std::size_t level, const auto& i, auto &)
        {
            level_field(level, i) = level;
            double dx = 1./(1<<level);
            auto coords = (xt::arange(i.start, i.end) + .5)*dx;
            u(level, i) = my_function(coords);
        });

        samurai::save("Step0_after", mesh, level_field, u);
    }
    

    return 0;
}
