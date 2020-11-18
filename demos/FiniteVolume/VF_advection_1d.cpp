#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"

#include <chrono>

double eps_g = 5.e-5, eps_f = 1e-1;

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
    samurai::BC<1> bc{ {{ {samurai::BCType::dirichlet, 1},
                       {samurai::BCType::dirichlet, 0},
                    }} };

    samurai::Field<Config> u{"u", mesh, bc};
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        double radius = .2;
        double x_center = 0;
        if (std::abs(center[0] - x_center) <= radius)
            u[cell] = 1;
        else
            u[cell] = 0;
    });

    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 4, max_level = 10;
    samurai::Box<double, dim> box({-2}, {2});
    samurai::Mesh<Config> mesh{box, min_level, max_level};

    double a = 1.;
    double dt = .95/(1<<max_level);

    auto u = init(mesh);

    spdlog::set_level(spdlog::level::warn);

    for (std::size_t nt=0; nt<500; ++nt)
    {
        std::cout << "iteration " << nt << "\n";
        tic();
        for (std::size_t i=0; i<max_level-min_level; ++i)
        {
            if (coarsening(u, i, nt))
                break;
        }
        auto duration = toc();
        std::cout << "coarsening: " << duration << "s\n";

        tic();
        for (std::size_t i=0; i<max_level-min_level; ++i)
        {
            if (refinement(u, i, nt))
                break;
        }
        duration = toc();
        std::cout << "refinement: " << duration << "s\n";

        samurai::mr_projection(u);
        samurai::amr_prediction(u);
        u.update_bc();

        tic();
        samurai::Field<Config> unp1{"u", mesh, u.bc()};
        unp1 = u - dt * samurai::upwind(a, u);


        for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

            stencil = {{-1}};

            auto subset_right = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                             mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_right([&](auto, auto& interval, auto)
            {
                auto i = interval[0];
                double dx = 1./(1<<level);

                // TODO: remove xt::eval bug. If we change the sign into the brackets, it works. Check why we can't set a minus at the first place.
                //       xtensor bug ??
                unp1(level, i) = unp1(level, i) - dt/dx * (-samurai::upwind_op<interval_t>(level, i).right_flux(a, u)
                                                           +samurai::upwind_op<interval_t>(level+1, 2*i+1).right_flux(a, u));
            });

            stencil = {{1}};

            auto subset_left = intersection(translate(mesh[samurai::MeshType::cells][level+1], stencil),
                                       mesh[samurai::MeshType::cells][level])
                               .on(level);

            subset_left([&](auto, auto& interval, auto)
            {
                auto i = interval[0];
                double dx = 1./(1<<level);

                unp1(level, i) = unp1(level, i) - dt/dx * (samurai::upwind_op<interval_t>(level, i).left_flux(a, u)
                                                          -samurai::upwind_op<interval_t>(level+1, 2*i).left_flux(a, u));
            });
        }

        std::swap(u.array(), unp1.array());

        duration = toc();
        std::cout << "upwind: " << duration << "s\n";

        tic();
        std::stringstream s;
        s << "VFadvection_1d_ite_" << nt;
        auto h5file = samurai::Hdf5(s.str().data());
        h5file.add_mesh(mesh);
        samurai::Field<Config> level_{"level", mesh};
        mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
        h5file.add_field(u);
        h5file.add_field(level_);
        duration = toc();
        std::cout << "save: " << duration << "s\n";
    }
    return 0;
}