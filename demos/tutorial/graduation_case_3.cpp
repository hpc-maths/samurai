#include <math.h>

#include <xtensor/xmasked_view.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

auto generate_mesh(std::size_t start_level)
{
    constexpr std::size_t dim = 2;
    samurai::Box<int, dim> box({-2<<start_level, -2<<start_level}, {2<<start_level, 2<<start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    return ca;
}

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t start_level = 1;
    std::size_t max_level = 6;
    bool with_graduation = true;
    auto ca = generate_mesh(start_level);

    std::size_t ite = 0;
    while(true)
    {
        std::cout << "Iteration for remove intersection: " << ite++ << "\n";

        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        samurai::for_each_cell(ca, [&](auto cell)
        {
            auto corner = cell.corner();
            double dx = cell.length;

            std::size_t npoints = 1<<(max_level+4);
            double dt = 2.*M_PI/npoints;
            double t = 0;

            for(std::size_t it = 0; it < npoints; ++it)
            {
                // double a = 1, b = 2, delta = 3.1415*.5;
                double a = 3, b = 2, delta = M_PI*.5;
                double xc = std::sin(a*t + delta);
                double yc = std::sin(b*t);

                if ((corner[0] < xc) && (corner[0] + dx > xc) &&
                    (corner[1] < yc) && (corner[1] + dx > yc))
                {
                    tag[cell] = true;
                    break;
                }
                t += dt;
            }
        });

        // graduation
        if (with_graduation)
        {
            xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

            for (std::size_t level = ca.max_level(); level > 1; --level)
            {
                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto subset = samurai::intersection(samurai::translate(ca[level], s), ca[level - 1]);

                    subset([&](const auto& interval, const auto& index)
                    {
                        auto j_f = index[0];
                        auto i_f = interval.even_elements();

                        if (i_f.is_valid())
                        {
                            auto mask = tag(level, i_f  - s[0], j_f - s[1]);
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;
                            xt::masked_view(tag(level - 1, i_c, j_c), mask) = true;
                        }

                        i_f = interval.odd_elements();
                        if (i_f.is_valid())
                        {
                            auto mask = tag(level, i_f  - s[0], j_f - s[1]);
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;
                            xt::masked_view(tag(level - 1, i_c, j_c), mask) = true;
                        }
                    });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index)
        {
            auto j = index[0];
            for (int i = interval.start; i < interval.end; ++i)
            {
                if (tag[i + interval.index] && level < max_level)
                {
                    cl[level + 1][{2*j}].add_interval({2*i, 2*i+2});
                    cl[level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
                }
                else
                {
                    cl[level][index].add_point(i);
                }
            }
        });

        samurai::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
    samurai::save("mesh_case3", ca);

    return 0;
}