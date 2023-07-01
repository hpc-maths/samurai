#include <iostream>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>

template <class Mesh>
auto init_field(Mesh& mesh, double dec)
{
    static constexpr std::size_t dim = Mesh::dim;
    auto f                           = samurai::make_field<double, 1>("u", mesh);
    f.fill(0.);
    samurai::for_each_cell(
        mesh,
        [&](const auto& cell)
        {
            auto x = cell.center(0);
            if constexpr (dim == 1)
            {
                if (x + dec > 0.4 && x + dec < 0.6)
                {
                    f[cell] = 1;
                }
            }
            else if constexpr (dim == 2)
            {
                auto y              = cell.center(1);
                const double radius = .2;
                if ((x - dec - .5) * (x - dec - .5) + (y - dec - .5) * (y - dec - .5) < radius * radius)
                {
                    f[cell] = 1;
                }
            }
            else if constexpr (dim == 3)
            {
                auto y              = cell.center(1);
                auto z              = cell.center(2);
                const double radius = .2;
                if ((x - dec - .5) * (x - dec - .5) + (y - dec - .5) * (y - dec - .5) + (z - dec - .5) * (z - dec - .5) < radius * radius)
                {
                    f[cell] = 1;
                }
            }
        });
    return f;
}

int main()
{
    constexpr std::size_t dim = 3;
    using Config              = samurai::MRConfig<dim>;
    auto box                  = samurai::Box<double, dim>({0., 0., 0.}, {1., 1., 1.});

    const std::size_t min_level = 2;
    const std::size_t max_level = 8;
    auto mesh1                  = samurai::MRMesh<Config>(box, min_level, max_level);
    auto f1                     = init_field(mesh1, 0);

    auto mesh2 = samurai::MRMesh<Config>(box, min_level, max_level);
    auto f2    = init_field(mesh2, 0.1);

    auto adapt_1 = samurai::make_MRAdapt(f1);
    auto adapt_2 = samurai::make_MRAdapt(f2);

    adapt_1(1e-3, 2);
    adapt_2(1e-3, 2);

    samurai::update_ghost_mr(f1);
    samurai::transfer(f1, f2);

    samurai::save("solution_src", mesh1, f1);
    samurai::save("solution_dst", mesh2, f2);

    return 0;
}
