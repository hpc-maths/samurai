#include <chrono>

#include <array>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/mr_config.hpp>
#include <samurai/mr/pred_and_proj.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/stencil_field.hpp>

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
samurai::Field<Config> init_u(samurai::Mesh<Config> &mesh)
{
    samurai::Field<Config> u("u", mesh);
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
    using Config = samurai::MRConfig<dim, 1, 1, 8>;

    double dt = .001;
    std::array<double, dim> a{1, 0};

    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::Mesh<Config> mesh{box, 5};

    // Initialization
    auto u = init_u(mesh);

    double eps = 1e-2;

    tic();
    for (std::size_t i = 0; i < 2; ++i)
    {
        samurai::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        samurai::mr_projection(u);
        samurai::mr_prediction(u);
        samurai::coarsening(detail, u, eps, i);
    }
    std::cout << "coarsening " << toc() << "\n";

    samurai::mr_projection(u);
    samurai::mr_prediction(u);

    auto h5file = samurai::Hdf5("advection_coarsening");
    h5file.add_mesh(mesh);
    samurai::Field<Config> level_{"level", mesh};
    mesh.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u);
    h5file.add_field(level_);

    // h5file.add_field_by_level(mesh, u);

    tic();
    for (std::size_t i = 0; i < 1; ++i)
    {
        samurai::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        samurai::mr_projection(u);
        samurai::mr_prediction(u);
        samurai::refinement(detail, u, eps);
    }
    std::cout << "refinement " << toc() << "\n";

    auto h5file_1 = samurai::Hdf5("advection_refinement");
    h5file_1.add_mesh(mesh);
    h5file_1.add_field(u);

    // // h5file_1.add_field_by_level(mesh, u);

    // // tic();
    // // for (std::size_t ite = 0; ite < 200; ++ite)
    // // {
    // //     mesh.make_projection(u);
    // //     mesh.make_prediction(u);
    // //     u = u - dt * samurai::upwind(a, u);

    // //     for (std::size_t i = 0; i < 10; ++i)
    // //     {
    // //         samurai::Field<Config> detail{"detail", mesh};
    // //         detail.array().fill(0);
    // //         mesh.make_projection(u);
    // //         mesh.make_prediction(u);
    // //         mesh.coarsening(detail, u, ite);
    // //     }

    // //     for (std::size_t i = 0; i < 1; ++i)
    // //     {
    // //         samurai::Field<Config> detail{"detail", mesh};
    // //         detail.array().fill(0);
    // //         mesh.make_projection(u);
    // //         mesh.make_prediction(u);
    // //         mesh.refinement(detail, u, ite);
    // //     }
    // // }
    // // std::cout << toc() << "\n";

    // // auto h5file_final = samurai::Hdf5("advection");
    // // h5file_final.add_mesh(mesh);
    // // h5file_final.add_field(u);
    return 0;
}