#include <chrono>

#include <array>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>
#include <mure/stencil_field.hpp>

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

template<class Config>
mure::Field<Config> init_u(mure::Mesh<Config> &mesh)
{
    mure::Field<Config> u("u", mesh);
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        double radius = .2;
        double x_center = 0., y_center = 0.;
        if (((center[0] - x_center) * (center[0] - x_center) +
             (center[1] - y_center) * (center[1] - y_center)) <=
            radius * radius)
            u[cell] = 1;
        else
            u[cell] = 0;
    });

    return u;
}

int main()
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;

    double dt = .01;
    std::array<double, dim> a{1, -1};

    mure::Box<double, dim> box({-1, -1}, {1, 1});
    mure::Mesh<Config> mesh{box, 5};

    // Initialization
    auto u = init_u(mesh);

    // tic();
    // for (std::size_t i = 0; i < 20; ++i)
    // {
    //     mure::Field<Config> detail{"detail", mesh};
    //     detail.array().fill(0);
    //     mesh.make_projection(u);
    //     mesh.coarsening(detail, u, i);
    // }
    // std::cout << toc() << "\n";

    tic();
    for (std::size_t ite = 0; ite < 50; ++ite)
    {
        // std::cout << "ite " << ite << "\n";
        u = u - dt * mure::upwind(a, u);
    }
    std::cout << toc() << "\n";

    auto h5file = mure::Hdf5("advection");
    h5file.add_mesh(mesh);
    h5file.add_field(u);
    return 0;
}