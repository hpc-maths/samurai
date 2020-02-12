#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mr/harten_multi.hpp>
#include <mure/mr/mesh.hpp>
#include <mure/mr/mr_config.hpp>
#include <mure/mr/pred_and_proj.hpp>

template <class Config>
auto init_u(mure::Mesh<Config> &mesh,
            double dx,
            std::size_t test_case)
{
    std::vector<mure::Field<Config>> u;
    u.push_back({"u", mesh});
    u[0].array().fill(0);

    switch (test_case)
    {
    case 1: //gaussian
        mesh.for_each_cell([&](auto &cell) {
            auto center = cell.center();
            auto x = center[0] - dx;
            auto y = center[1] - dx;
            u[0][cell] = exp(-20 * (x * x + y * y));
        });
        break;
    case 2: //diamond
        mesh.for_each_cell([&](auto &cell) {
            auto center = cell.center();
            double theta = M_PI / 4;
            auto x = cos(theta) * center[0] - sin(theta) * center[1];
            auto y = sin(theta) * center[0] + cos(theta) * center[1];
            double x_corner = -(0.1 - dx);
            double y_corner = -(0.1 - dx);
            double length = 0.2;

            if ((x_corner <= x) and (x <= x_corner + length) and 
                (y_corner <= y) and (y <= y_corner + length))
                u[0][cell] = 1;
            else
                u[0][cell] = 0;
        });
        break;
    case 3: //circle
        mesh.for_each_cell([&](auto &cell) {
            auto center = cell.center();
            double radius = .2;
            double x_center = dx, y_center = dx;
            if (((center[0] - x_center) * (center[0] - x_center) + 
                 (center[1] - y_center) * (center[1] - y_center))
                 <= radius * radius)
                u[0][cell] = 1;
            else
                u[0][cell] = 0;
        });
        break;
    default:
        std::cout << "unknown test case !!\n";
    }

    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;

    std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};

    cxxopts::Options options("adv_ana_2d", "Analytic advection using multi resolution for 2D problems");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("8"))
                       ("test", "test case (1: gaussian, 2: diamond, 3: circle)", cxxopts::value<std::size_t>()->default_value("1"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>(),
                        max_level = result["max_level"].as<std::size_t>();
            std::size_t test_case = result["test"].as<std::size_t>();
            double eps = 1e-2;
            double dx = 1. / (1 << max_level);
            std::cout << "dx = " << dx << "\n";

            mure::Box<double, dim> box({-2, -2}, {2, 2});
            mure::Mesh<Config> mesh{box, max_level};

            for (std::size_t ite = 0; ite < 100; ++ite)
            {
                std::cout << "iteration: " << ite << "\n";
                auto u = init_u(mesh, ite * dx, test_case);

                for (std::size_t i = 0; i < max_level - min_level; ++i)
                {
                    mure::mr_projection(u[0]);
                    mure::mr_prediction(u[0]);
                    mure::harten_multi(u, eps, i, min_level);
                }

                std::stringstream s;
                s << "advection_" << ite;
                auto h5file = mure::Hdf5(s.str().data());
                h5file.add_mesh(mesh);
                mure::Field<Config> level_{"level", mesh};
                mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
                h5file.add_field(u[0]);
                h5file.add_field(level_);
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}