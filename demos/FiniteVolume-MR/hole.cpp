#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"

template<class Config>
auto build_mesh(std::size_t min_level, std::size_t max_level)
{
    constexpr std::size_t dim = Config::dim;

    mure::Box<double, dim> box({0, 0}, {1, 1});
    mure::Mesh<Config> mesh{box, min_level, max_level};

    mure::CellList<Config> cl;
    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                > radius * radius)
        {
            cl[cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
        }
    });
    mure::Mesh<Config> new_mesh(cl, min_level, max_level);
    return new_mesh;
}

template<class Config>
auto init_f(mure::Mesh<Config> &mesh)
{
    mure::BC<2> bc{ {{ {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0}
                    }} };

    mure::Field<Config> f("f", mesh, bc);
    f.array().fill(1);

    return f;
}

int main()
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim, 2>;

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

    auto h5file = mure::Hdf5("hole");
    h5file.add_mesh(mesh);
    return 0;
}