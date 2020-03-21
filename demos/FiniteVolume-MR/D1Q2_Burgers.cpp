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

        double v = .5 * u * u;

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
    });

    return f;
}

template<class Field>
void one_time_step_old(Field &f)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1., s = 1.;
    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    std::cout << max_level << "\n";

    mure::mr_projection(f);
    mure::mr_prediction(f);

    Field new_f{"f_new", mesh};
    new_f.array().fill(0);

    std::array<std::array<double, 5>, 15> cp = {
        {{{-1. / 8, 9. / 8, -7. / 8, -1. / 8, 0}},
         {{-9. / 64, 33. / 32, -5. / 8, -9. / 32, 1. / 64}},
         {{-33. / 256, 117. / 128, -13. / 32, -53. / 128, 9. / 256}},
         {{-117. / 1024, 421. / 512, -1. / 4, -261. / 512, 53. / 1024}},
         {{-421. / 4096, 1557. / 2048, -19. / 128, -1173. / 2048, 261. / 4096}},
         {{-1557. / 16384, 5909. / 8192, -11. / 128, -5013. / 8192,
           1173. / 16384}},
         {{-5909. / 65536, 22869. / 32768, -25. / 512, -20821. / 32768,
           5013. / 65536}},
         {{-22869. / 262144, 89685. / 131072, -7. / 256, -85077. / 131072,
           20821. / 262144}},
         {{-89685. / 1048576, 354645. / 524288, -31. / 2048, -344405. / 524288,
           85077. / 1048576}},
         {{-354645. / 4194304, 1409365. / 2097152, -17. / 2048,
           -1386837. / 2097152, 344405. / 4194304}},
         {{-1409365. / 16777216, 5616981. / 8388608, -37. / 8192,
           -5567829. / 8388608, 1386837. / 16777216}},
         {{-5616981. / 67108864, 22422869. / 33554432, -5. / 2048,
           -22316373. / 33554432, 5567829. / 67108864}},
         {{-22422869. / 268435456, 89593173. / 134217728, -43. / 32768,
           -89363797. / 134217728, 22316373. / 268435456}},
         {{-89593173. / 1073741824, 358159701. / 536870912, -23. / 32768,
           -357668181. / 536870912, 89363797. / 1073741824}},
         {{-358159701. / 4294967296, 1432180053. / 2147483648, -49. / 131072,
           -1431131477. / 2147483648, 357668181. / 4294967296}}}};

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];

            auto fp = f(0, level, i - 1);
            auto fm = f(1, level, i + 1);

            if (level != max_level)
            {
                std::size_t j = max_level - level;
                double coeff = 1. / (1 << j);

                auto cpp = cp[j - 1][0] * f(0, level, i - 2)+
                           cp[j - 1][1] * f(0, level, i - 1)+
                           cp[j - 1][2] * f(0, level, i )+
                           cp[j - 1][3] * f(0, level, i + 1)+
                           cp[j - 1][4] * f(0, level, i + 2);

                fp = f(0, level, i) + coeff * cpp;

                auto cmm = cp[j - 1][4] * f(1, level, i - 2)+
                           cp[j - 1][3] * f(1, level, i - 1)+
                           cp[j - 1][2] * f(1, level, i )+
                           cp[j - 1][1] * f(1, level, i + 1)+
                           cp[j - 1][0] * f(1, level, i + 2);

                fm = f(1, level, i) + coeff * cmm;
            }

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            vv = (1 - s) * vv + s * .5 * uu * uu;

            new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}

template<class Field, class interval_t>
auto prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item)
{
    auto step = i.step;
    auto ig = i / 2;
    ig.step = step >> 1;

    xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});
    for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
    {
        d[iii] = (ii & 1)? -1.: 1.;
    }

    if (level == 1)
    {
        return xt::eval(f(item, level_g, ig) - 1./8  * d * (f(item, level_g, ig+1)
                                                          - f(item, level_g, ig-1)));
    }
    return xt::eval(prediction(f, level_g, level-1, ig, item) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item) 
                                                                          - prediction(f, level_g, level-1, ig-1, item)));
}

template<class Field>
void one_time_step(Field &f)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1., s = 1.;
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

            auto fp = f(0, level, i - 1);
            auto fm = f(1, level, i + 1);

            if (level != max_level)
            {
                std::size_t j = max_level - level;
                double coeff_p = 1. / (1 << level);
                double coeff_m = 1. / (1 << j);

                fp = f(0, level, i) + coeff_p * (prediction(f, level, j, i*(1<<j)-1, 0)
                                               - prediction(f, level, j, (i+1)*(1<<j)-1, 0));

                fm = f(1, level, i) - coeff_m * (prediction(f, level, j, i*(1<<j), 1)
                                               - prediction(f, level, j, (i+1)*(1<<j), 1));
            }

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            vv = (1 - s) * vv + s * .5 * uu * uu;

            new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}

template<class Field>
void save_solution(Field &f, double eps, std::size_t ite)
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D1Q2_Burders_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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

            // Initialization
            auto f = init_f(mesh, 0);

            for (std::size_t nb_ite = 0; nb_ite < 100; ++nb_ite)
            {
                std::cout << nb_ite << "\n";

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

                one_time_step(f);

                save_solution(f, eps, nb_ite);
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
