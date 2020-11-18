#include <array>
#include <math.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/mr_config.hpp>
#include <samurai/mr/pred_and_proj.hpp>

template<class Config>
samurai::Field<Config> init_u(samurai::Mesh<Config> &mesh, int test_case)
{
    samurai::Field<Config> u("u", mesh);
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        switch (test_case)
        {
        case 1:
            u[cell] = exp(-50.0 * x * x);
            break;
        case 2:
            u[cell] = 1 - sqrt(abs(sin(M_PI / 2 * x)));
            break;
        case 3:
            u[cell] = 1 - tanh(50.0 * abs(x));
            break;
        case 4:
            u[cell] = 0.5 - abs(x);
            break;
        case 5:
            u[cell] = sin(x) * (tanh(50.0 * x) + tanh(50.0 * (x - 0.5)));
            break;
        default:
            u[cell] = exp(-50.0 * x * x);
            break;
        }
    });

    return u;
}

template<class Config>
samurai::Field<Config> compute_error(samurai::Mesh<Config> &mesh,
                                  samurai::Field<Config> solution, int test_case)
{
    samurai::Field<Config> error("error", mesh);
    error.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        switch (test_case)
        {
        case 1:
            error[cell] = abs(exp(-50.0 * x * x) - solution[cell]);
            break;
        case 2:
            error[cell] =
                abs(1 - sqrt(abs(sin(M_PI / 2 * x))) - solution[cell]);
            break;
        case 3:
            error[cell] = abs(1 - tanh(50.0 * abs(x)) - solution[cell]);
            break;
        case 4:
            error[cell] = abs(0.5 - abs(x) - solution[cell]);
            break;
        case 5:
            error[cell] =
                abs(sin(x) * (tanh(50.0 * x) + tanh(50.0 * (x - 0.5))) -
                    solution[cell]);
            break;
        default:
            error[cell] = abs(exp(-50.0 * x * x) - solution[cell]);
            break;
        }
    });
    return error;
}

// The input arguments are supposed to be the following:
// test case | maximum level | eps
int main(int argc, char *argv[])
{

    if (argc <= 3)
    {
        std::cerr << "Provide the number of the test case, the maximum level "
                     "and the epsilon, please!"
                  << std::endl;
        return -1;
    }

    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({-1}, {1});
    samurai::Mesh<Config> mesh{box, atoi(argv[2])};

    // Initialization
    auto u = init_u(mesh, atoi(argv[1]));

    // double eps = 1e-2;
    double eps = atof(argv[3]);

    for (std::size_t i = 0; i < 20; ++i)
    {
        // if (i == 3)
        // std::cout << "#################################\n";
        // std::cout << mesh << "\n";
        // std::cout << "#################################\n";
        samurai::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        samurai::mr_projection(u);
        samurai::coarsening(detail, u, eps, i);
    }

    // Error computation
    auto error = compute_error(mesh, u, atoi(argv[1]));

    std::stringstream str;
    str << "coarsening1d_case-" << argv[1] << "_lmax-" << argv[2] << "_eps-"
        << argv[3];
    auto h5file = samurai::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    samurai::Field<Config> level_{"level", mesh};
    mesh.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u);
    h5file.add_field(level_);
    h5file.add_field(error);

    return 0;
}
