#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"

template<class Config>
auto build_mesh(std::size_t min_level, std::size_t max_level)
{
    constexpr std::size_t dim = Config::dim;

    samurai::Box<double, dim> box({0, 0}, {2, 1});
    samurai::Mesh<Config> mesh{box, min_level, max_level};

    samurai::CellList<Config> cl;
    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double radius = 1./32.;
        double x_center = 5./16. + radius, y_center = 0.5;

        if ((   std::max(std::abs(x - x_center),
                std::abs(y - y_center)))
                > radius)
        {
            cl[cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
        }
    });
    samurai::Mesh<Config> new_mesh(cl, min_level, max_level);
    return new_mesh;
}

template<class Config>
auto init_f(samurai::Mesh<Config> &mesh)
{
    samurai::BC<2> bc{ {{ {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0}
                    }} };

    samurai::Field<Config> f("f", mesh, bc);
    f.array().fill(1);

    return f;
}

int main()
{
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim, 2>;

    std::size_t min_level = 2;
    std::size_t max_level = 10;
    double eps = 1e-4;

    auto mesh = build_mesh<Config>(min_level, max_level);

    auto f = init_f(mesh);

    for (std::size_t i=0; i<max_level-min_level; ++i)
    {
        if (coarsening(f, eps, i))
            break;
    }


    for (std::size_t i=0; i<max_level-min_level; ++i)
    {
        if (refinement(f, eps, 0.0, i))
            break;
    }

    auto h5file = samurai::Hdf5("hole");
    h5file.add_mesh(mesh);

    {
        std::stringstream str;
        str << "hole_level_by_level";
        auto h5file = samurai::Hdf5(str.str().data());
        h5file.add_field_by_level(mesh, f);
    }

    return 0;
}
