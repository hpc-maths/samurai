#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"

double lambda = 2.;
double sigma_q = 0.5; 
double sigma_xy = 0.5;

double sq = 1.2;//1./(.5 + sigma_q);
double sxy = 1./(.5 + sigma_xy);

double kx = 0.0;
double ky = 1.5;

template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 4;
    mure::BC<2> bc{ {{ {mure::BCType::dirichlet, 1},
                       {mure::BCType::dirichlet, 1},
                       {mure::BCType::dirichlet, 1},
                       {mure::BCType::dirichlet, 1}
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double m0 = 0;

        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                <= radius * radius)
            m0 = 1;

        double m1 = 0.5 * kx * m0 * m0;
        double m2 = 0.5 * ky * m0 * m0;
        double m3 = 0.0;

        // We come back to the distributions
        f[cell][0] = .25 * m0 + .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
        f[cell][1] = .25 * m0                    + .5/lambda * (m2) - .25/(lambda*lambda) * m3;
        f[cell][2] = .25 * m0 - .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
        f[cell][3] = .25 * m0                    - .5/lambda * (m2) - .25/(lambda*lambda) * m3;

    });

    return f;
}

template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(4);
        for (int l = 0; l < size; ++l)
        {
            data[k][0] += prediction(k, i*size - 1, j*size + l) - prediction(k, (i+1)*size - 1, j*size + l);
            data[k][1] += prediction(k, i*size + l, j*size - 1) - prediction(k, i*size + l, (j+1)*size - 1);
            data[k][2] += prediction(k, (i+1)*size, j*size + l) - prediction(k, i*size, j*size + l);
            data[k][3] += prediction(k, i*size + l, (j+1)*size) - prediction(k, i*size + l, j*size);
        }
    }
    return data;
}

template<class Field, class pred>
void one_time_step(Field &f, const pred& pred_coeff)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    mure::mr_prediction(f);

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto& index, auto &interval, auto) {
            auto k = interval[0]; // Logical index in x
            auto h = index[0];    // Logical index in y

            std::size_t j = max_level - level; 
            double coeff = 1. / (1 << (2*j)); // The factor 2 comes from the 2D 

            auto f0 = xt::eval(f(0, level, k, h));
            auto f1 = xt::eval(f(1, level, k, h));
            auto f2 = xt::eval(f(2, level, k, h));
            auto f3 = xt::eval(f(3, level, k, h));

            // We have to iterate over the elements on the considered boundary
            for(auto &c: pred_coeff[j][0].coeff)
            {
                coord_index_t stencil_x, stencil_y;
                std::tie(stencil_x, stencil_y) = c.first;
                f0 += coeff*c.second*f(0, level, k + stencil_x, h + stencil_y);
            }

            for(auto &c: pred_coeff[j][1].coeff)
            {
                coord_index_t stencil_x, stencil_y;
                std::tie(stencil_x, stencil_y) = c.first;
                f1 += coeff*c.second*f(1, level, k + stencil_x, h + stencil_y);
            }

            for(auto &c: pred_coeff[j][2].coeff)
            {
                coord_index_t stencil_x, stencil_y;
                std::tie(stencil_x, stencil_y) = c.first;
                f2 += coeff*c.second*f(2, level, k + stencil_x, h + stencil_y);
            }

            for(auto &c: pred_coeff[j][3].coeff)
            {
                coord_index_t stencil_x, stencil_y;
                std::tie(stencil_x, stencil_y) = c.first;
                f3 += coeff*c.second*f(3, level, k + stencil_x, h + stencil_y);
            }

            // We compute the advected momenti
            auto m0 = xt::eval(                 f0 + f1 + f2 + f3) ;
            auto m1 = xt::eval(lambda        * (f0      - f2      ));
            auto m2 = xt::eval(lambda        * (     f1      - f3));
            auto m3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));

            m1 = (1 - sq) * m1 + sq * 0.5 * kx * m0 * m0;
            m2 = (1 - sq) * m2 + sq * 0.5 * ky * m0 * m0;
            m3 = (1 - sxy) * m3; 

            // We come back to the distributions
            new_f(0, level, k, h) = .25 * m0 + .5/lambda * m1                    + .25/(lambda*lambda) * m3;
            new_f(1, level, k, h) = .25 * m0                    + .5/lambda * m2 - .25/(lambda*lambda) * m3;
            new_f(2, level, k, h) = .25 * m0 - .5/lambda * m1                    + .25/(lambda*lambda) * m3;
            new_f(3, level, k, h) = .25 * m0                    - .5/lambda * m2 - .25/(lambda*lambda) * m3;
        });
    }

    std::swap(f.array(), new_f.array());
}

template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q4_burgers_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> u{"u", mesh};
    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        u[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
    });
    h5file.add_field(u);
    h5file.add_field(f);
    h5file.add_field(level_);
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_scalar_burgers",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar burgers equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("10"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 2;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();

            mure::Box<double, dim> box({0, 0}, {1, 1});
            mure::Mesh<Config> mesh{box, min_level, max_level};

            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

            // Initialization
            auto f = init_f(mesh, 0);

            double T = 1.2;
            double dx = 1.0 / (1 << max_level);
            double dt = dx;

            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout << nb_ite << "\n";

                if (nb_ite > 0)
                    save_solution(f, eps, nb_ite);

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (coarsening(f, eps, i))
                        break;
                }

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (refinement(f, eps, i))
                        break;
                }

                f.update_bc();

                if (nb_ite == 0)    {

                    std::stringstream str;
                    str << "debug_BG";

                    auto h5file = mure::Hdf5(str.str().data());
                    h5file.add_mesh(mesh);
                    // We save with the levels
                    h5file.add_field_by_level(mesh, f);

                }

                one_time_step(f, pred_coeff);
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
