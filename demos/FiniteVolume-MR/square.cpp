#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "../FiniteVolume/refinement.hpp"
#include "criteria.hpp"
#include "evolve_mesh.hpp"

#include <chrono>


/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template <class Config>
auto init(samurai::Mesh<Config> &mesh)
{
    samurai::BC<2> bc{ {{ {samurai::BCType::dirichlet, 1},
                       {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0},
                       {samurai::BCType::dirichlet, 0}
                    }} };
    samurai::Field<Config> u{"u", mesh, bc};
    u.array().fill(0);


    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     double len = .1;
    //     double x_center = 0.5, y_center = 0.5;
    //     if (std::abs(center[0] - x_center) < len
    //             and std::abs(center[1] - y_center) < len)
    //         u[cell] = 1;
    //     else
    //         u[cell] = 0;
    // });


    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if (((center[0] - x_center) * (center[0] - x_center) +
                (center[1] - y_center) * (center[1] - y_center))
                <= radius * radius)
            u[cell] = 1;
        else
            u[cell] = 0;
    });

    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     auto x = center[0] - 0.5;
    //     auto y = center[1] - 0.5;
    //     u[cell] = exp(-500.0 * (x * x + y * y));
    // });


    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 2, max_level = 12;
    // samurai::Box<double, dim> box({-2, -2}, {2, 2});
    samurai::Box<double, dim> box({0, 0}, {1, 1});

    samurai::Mesh<Config> mesh1{box, min_level, max_level};
    samurai::Mesh<Config> mesh2{box, min_level, max_level};


    auto u1 = init(mesh1);
    auto u2 = init(mesh2);

    spdlog::set_level(spdlog::level::warn);


    //    double max_det = max_detail(u);
    double eps = 1.0e-1;

    tic();
    for (std::size_t i = 0; i < max_level - min_level; ++i)
    {
        if (coarsening(u1, eps, i))
            break;
    }

    for (std::size_t i = 0; i < max_level - min_level; ++i)
    {
        //if(refinement(u1, eps, i))
        //    break;
    }


    auto duration = toc();
    std::cout<<"\nTime separated procedure = "<<duration;

    {
        std::stringstream s;
        s << "after_separated";
        auto h5file = samurai::Hdf5(s.str().data());
        h5file.add_mesh(mesh1);
        h5file.add_field(u1);
    }

    tic();
    for (std::size_t i = 0; i < max_level - min_level; ++i) {
        if(evolve_mesh(u2, eps, i))
            break;
    }
    duration = toc();
    std::cout<<"\nTime merged procedure = "<<duration<<std::endl;

    {
        std::stringstream s;
        s << "after_merged";
        auto h5file = samurai::Hdf5(s.str().data());
        h5file.add_mesh(mesh2);
        h5file.add_field(u2);
    }
    return 0;
}