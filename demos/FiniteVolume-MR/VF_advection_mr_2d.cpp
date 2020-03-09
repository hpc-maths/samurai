#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "../FiniteVolume/refinement.hpp"
#include "criteria.hpp"

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
auto init(mure::Mesh<Config> &mesh)
{
    mure::BC<2> bc{ {{ {mure::BCType::dirichlet, 1},
                       {mure::BCType::dirichlet, 0},
                       {mure::BCType::neumann, -0.5},
                       {mure::BCType::neumann, 0}
                    }} };
    mure::Field<Config> u{"u", mesh, bc};
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0] - 0.5;
        auto y = center[1] - 0.5;
        u[cell] = exp(-150.0 * (x * x + y * y));
    });
    /*
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
    */
    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     double theta = M_PI / 4;
    //     auto x = cos(theta) * center[0] - sin(theta) * center[1];
    //     auto y = sin(theta) * center[0] + cos(theta) * center[1];
    //     double x_corner = -0.1;
    //     double y_corner = -0.1;
    //     double length = 0.2;

    //     if ((x_corner <= x) and (x <= x_corner + length) and 
    //         (y_corner <= y) and (y <= y_corner + length))
    //         u[cell] = 1;
    //     else
    //         u[cell] = 0;
    // });

    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 2, max_level = 9;
    // mure::Box<double, dim> box({-2, -2}, {2, 2});
    mure::Box<double, dim> box({0, 0}, {1, 1});
    mure::Mesh<Config> mesh{box, min_level, max_level};

    std::array<double, 2> a{{1, 1}};
    double dt = .5/(1<<max_level);

    auto u = init(mesh);

    spdlog::set_level(spdlog::level::warn);


    // Ca ne fait pas ce quon lui demande. Il le fait a peine
    // pour le cercle mais pas du tout pour la gaussienne
    // Il faut faire de l'affichage des details

    // A faire demain

    coarsening(u);

    std::stringstream s;
    s << "VF-advection_mr_2d";
    auto h5file = mure::Hdf5(s.str().data());
    h5file.add_mesh(mesh);
    h5file.add_field(u);
        
    return 0;
}