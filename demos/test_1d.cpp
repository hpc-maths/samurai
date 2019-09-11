#include <array>
#include <math.h>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mr/coarsening.hpp>
#include <mure/mr/mesh.hpp>
#include <mure/mr/mr_config.hpp>
#include <mure/mr/pred_and_proj.hpp>

template<class Config>
mure::Field<Config> init_u(mure::Mesh<Config> &mesh, int test_case)
{
    mure::Field<Config> u("u", mesh);
    u.array().fill(0);
    
    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        switch(test_case)   {
            case 1: u[cell] = exp(-50.0*x*x); break;
            case 2: u[cell] = 1-sqrt(abs(sin(M_PI/2*x))); break;
            case 3: u[cell] = tanh(50.0*x); break;
            case 4: u[cell] = 0.5-abs(x); break;
            case 5: u[cell] = sin(x)*(tanh(50.0*x) + tanh(50.0*(x-0.5))); break;
            default: u[cell] = exp(-50.0*x*x); break;
        }
    });

    return u;
}

template<class Config>
mure::Field<Config> compute_error(mure::Mesh<Config> &mesh, mure::Field<Config> solution, int test_case)
{
    mure::Field<Config> error("error", mesh);
    error.array().fill(0);   

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        switch(test_case)   {
            case 1: error[cell] = abs(exp(-50.0*x*x)-solution[cell]); break;
            case 2: error[cell] = abs(1-sqrt(abs(sin(M_PI/2*x)))-solution[cell]); break;
            case 3: error[cell] = abs(tanh(50.0*x)-solution[cell]); break;
            case 4: error[cell] = abs(0.5-abs(x)-solution[cell]); break;
            case 5: error[cell] = abs(sin(x)*(tanh(50.0*x) + tanh(50.0*(x-0.5))) - solution[cell]); break;
            default: error[cell] = abs(exp(-50.0*x*x)-solution[cell]); break;
        }
    });
    return error;

}



// The input arguments are supposed to be the following:
// test case | maximum level | eps
int main(int argc, char *argv[])
{

    if (argc <= 3)  {
        std::cerr<<"Provide the number of the test case, the maximum level and the epsilon, please!"<<std::endl;
        return -1;
    }

    constexpr size_t dim = 1;
    using Config = mure::MRConfig<dim>;

    mure::Box<double, dim> box({-1}, {1});
    mure::Mesh<Config> mesh{box, atoi(argv[2])};

    // Initialization
    auto u = init_u(mesh, atoi(argv[1]));

    //double eps = 1e-2;
    double eps = atof(argv[3]);

    for (std::size_t i = 0; i < 20; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        mure::mr_projection(u);
        mure::coarsening(detail, u, eps, i);
    }

    // Error computation
    auto error = compute_error(mesh, u, atoi(argv[1]));

    auto h5file = mure::Hdf5("test_1d");
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mesh.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u);
    h5file.add_field(level_);
    h5file.add_field(error);

    return 0;
}
