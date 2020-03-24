#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"

template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 2;
    mure::BC<1> bc{ {{ {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0},
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        double u = 0;

        if (x >= -1 and x < t)
        {
            u = (1 + x) / (1 + t);
        }
        if (x >= t and x < 1)
        {
            u = (1 - x) / (1 - t);
        }

        //double u = exp(-20.0 * x * x);

        //double v = .5 * u; 
        double v = .5 * u * u;

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
    });

    return f;
}

template<class Field, class interval_t>
auto prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item)
{
    if (level == 0)
    {
        return xt::eval(f(item, level_g, i));
    }

    auto step = i.step;
    auto ig = i / 2;
    ig.step = step >> 1;
    xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

    for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
    {
        d[iii] = (ii & 1)? -1.: 1.;
    }
  
    return xt::eval(prediction(f, level_g, level-1, ig, item) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item) 
                                                                          - prediction(f, level_g, level-1, ig-1, item)));
}

template<class Field>
void one_time_step(Field &f)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1., s = 1.0;
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
        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];

            // auto fp = f(0, level, i - 1);
            // auto fm = f(1, level, i + 1);

            // if (level != max_level)
            // {
            //     std::size_t j = max_level - level;
            //     double coeff = 1. / (1 << j);

            //     // std::cout << "interval " << i << " j " << j << "\n";
            //     // std::cout << "calcul fp\n";
            //     fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0)
            //                                  - prediction(f, level, j, (i+1)*(1<<j)-1, 0));

            //     // std::cout << "calcul fm\n";
            //     fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1)
            //                                  - prediction(f, level, j, (i+1)*(1<<j), 1));
            // }

            std::size_t j = max_level - level;
            double coeff = 1. / (1 << j);
            auto fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0)
                                            - prediction(f, level, j, (i+1)*(1<<j)-1, 0));

            // std::cout << "calcul fm\n";
            auto fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1)
                                            - prediction(f, level, j, (i+1)*(1<<j), 1));

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            vv = (1 - s) * vv + s * .5 * uu * uu;
            //vv = (1 - s) * vv + s * .5 * uu;

            new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}

template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext)
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D1Q2_Burgers_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> u{"u", mesh};
    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        u[cell] = f[cell][0] + f[cell][1];
    });
    h5file.add_field(u);
    h5file.add_field(f);
    h5file.add_field(level_);
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers",
                             "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

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
            constexpr size_t dim = 1;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();

            mure::Box<double, dim> box({-3}, {3});
            mure::Mesh<Config> mesh{box, min_level, max_level};
            // mure::Mesh<Config> mesh_old{box, min_level, max_level};

            // mure::CellList<Config> cl;
            // cl[6][{}].add_interval({-192, 0});
            // cl[5][{}].add_interval({0, 96});
            // mure::Mesh<Config> mesh{cl, mesh_old.initial_mesh(), min_level, max_level};

            // Initialization
            auto f = init_f(mesh, 0);

            double T = 1.2;
            double dx = 1.0 / (1 << max_level);
            double dt = dx;

            std::size_t N = static_cast<std::size_t>(T / dt);


            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout << nb_ite << "\n";

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (coarsening(f, eps, i))
                        break;
                }

                save_solution(f, eps, nb_ite, "coarsening");

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (refinement(f, eps, i))
                        break;
                }

                save_solution(f, eps, nb_ite, "refinement");

                one_time_step(f);

                save_solution(f, eps, nb_ite, "onetimestep");
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
