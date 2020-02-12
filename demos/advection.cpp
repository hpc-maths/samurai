#include <chrono>

#include <array>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mr/coarsening.hpp>
#include <mure/mr/mesh.hpp>
#include <mure/mr/mr_config.hpp>
#include <mure/mr/pred_and_proj.hpp>
#include <mure/mr/refinement.hpp>
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

    // mesh.for_each_cell([&](auto &cell) {
    //     auto center = cell.center();
    //     double radius = .2;
    //     double x_center = .65, y_center = -.5;
    //     // bug pour le cas du dessous avec 10 niveaux (entre ite 8 et 9)
    //     if (((center[0] - x_center) * (center[0] - x_center) +
    //          (center[1] - y_center) * (center[1] - y_center)) <=
    //         radius * radius)
    //         // if ((center[0]*center[0] + center[1]*center[1]) <= 0.25*0.25)
    //         u[cell] = 1;
    //     else
    //         u[cell] = 0;
    // });

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
    using Config = mure::MRConfig<dim, 1, 1, 8>;

    double dt = .001;
    std::array<double, dim> a{1, 0};

    mure::Box<double, dim> box({-1, -1}, {1, 1});
    mure::Mesh<Config> mesh{box, 5};

    // Initialization
    auto u = init_u(mesh);

    double eps = 1e-2;

    tic();
    for (std::size_t i = 0; i < 2; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        mure::mr_projection(u);
        mure::mr_prediction(u);
        mure::coarsening(detail, u, eps, i);
    }
    std::cout << "coarsening " << toc() << "\n";

    mure::mr_projection(u);
    mure::mr_prediction(u);

    auto h5file = mure::Hdf5("advection_coarsening");
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mesh.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u);
    h5file.add_field(level_);

    // h5file.add_field_by_level(mesh, u);

    tic();
    for (std::size_t i = 0; i < 1; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        mure::mr_projection(u);
        mure::mr_prediction(u);
        mure::refinement(detail, u, eps);
    }
    std::cout << "refinement " << toc() << "\n";

    auto h5file_1 = mure::Hdf5("advection_refinement");
    h5file_1.add_mesh(mesh);
    h5file_1.add_field(u);

    // // h5file_1.add_field_by_level(mesh, u);

    // // tic();
    // // for (std::size_t ite = 0; ite < 200; ++ite)
    // // {
    // //     mesh.make_projection(u);
    // //     mesh.make_prediction(u);
    // //     u = u - dt * mure::upwind(a, u);

    // //     for (std::size_t i = 0; i < 10; ++i)
    // //     {
    // //         mure::Field<Config> detail{"detail", mesh};
    // //         detail.array().fill(0);
    // //         mesh.make_projection(u);
    // //         mesh.make_prediction(u);
    // //         mesh.coarsening(detail, u, ite);
    // //     }

    // //     for (std::size_t i = 0; i < 1; ++i)
    // //     {
    // //         mure::Field<Config> detail{"detail", mesh};
    // //         detail.array().fill(0);
    // //         mesh.make_projection(u);
    // //         mesh.make_prediction(u);
    // //         mesh.refinement(detail, u, ite);
    // //     }
    // // }
    // // std::cout << toc() << "\n";

    // // auto h5file_final = mure::Hdf5("advection");
    // // h5file_final.add_mesh(mesh);
    // // h5file_final.add_field(u);
    return 0;
}