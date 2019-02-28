#include <chrono>
// #include <math.h>
#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>

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

int main()
{
    constexpr size_t dim = 1;
    using Config = mure::MRConfig<dim>;

    mure::Box<double, dim> box({-1, -1, -1}, {1, 1, 1});

    mure::Mesh<Config> mesh{box, 7};
    mure::Field<Config> u{"u", mesh};

    // mesh.for_each_cell([&](auto& cell){
    //     auto center = cell.center();
    //     if (center[0] >= .4 and center[0] <= .6)
    //         u[cell] = 1;
    //     else
    //         u[cell] = 0;
    // });

    // mesh.for_each_cell([&](auto& cell){
    //     auto center = cell.center();
    //     u[cell] = exp(-50*center[0]*center[0]) - exp(-50);
    // });

    // mesh.for_each_cell([&](auto& cell){
    //     auto center = cell.center();
    //     u[cell] = tanh(50*fabs(center[0])) - 1;
    // });

    mesh.for_each_cell([&](auto& cell){
        auto center = cell.center();
        u[cell] = 1 - sqrt(fabs(sin(M_PI_2*center[0])));
    });

    for (std::size_t i=0; i<10; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        mesh.projection(u);
        mesh.detail(detail, u);
    }

    mure::Field<Config> level_{"level", mesh};
    mesh.for_each_cell([&](auto& cell){
        level_[cell] = cell.level;
    });
    std::cout << u << "\n";

    auto h5file = mure::Hdf5("heaviside");
    h5file.add_mesh(mesh);
    h5file.add_field(u);
    h5file.add_field(level_);
    return 0;
}