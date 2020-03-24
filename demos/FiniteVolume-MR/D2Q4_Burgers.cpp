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
    constexpr std::size_t nvel = 4;
    mure::BC<2> bc{ {{ {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0},
                       {mure::BCType::dirichlet, 0}
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double m0 = 0;

        double radius = .5;
        double x_center = 0.0, y_center = 0.0;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                <= radius * radius)
            m0 = 1;


        double lambda = 1.0; // Just here
        double angle = M_PI / 4.0;
        double kx = cos(angle);
        double ky = sin(angle);

        double m1 = kx*m0*m0/2.0;
        double m2 = ky*m0*m0/2.0;
        double m3 = m0/2.0;  // We can change this but normally it works fine

        // We come back to the distributions
        f[cell][0] = .25 * m0 + .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
        f[cell][1] = .25 * m0                    + .5/lambda * (m2) - .25/(lambda*lambda) * m3;
        f[cell][2] = .25 * m0 - .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
        f[cell][3] = .25 * m0                    - .5/lambda * (m2) - .25/(lambda*lambda) * m3;

    });

    return f;
}

template<class Field, class interval_t, class index_t>
auto prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &k, const index_t h, const std::size_t item)
{
    if (level == 0)
    {
        return xt::eval(f(item, level_g, k, h));
    }

    auto step = k.step;
    auto kg = k / 2;
    auto hg = h / 2;
    kg.step = step >> 1;
    xt::xtensor<double, 1> d_x = xt::empty<double>({k.size()/k.step});
    xt::xtensor<double, 1> d_xy = xt::empty<double>({k.size()/k.step});
    double d_y = (h & 1)? -1.: 1.;

    for (int ii=k.start, iii=0; ii<k.end; ii+=k.step, ++iii)
    {
        d_x[iii] = (ii & 1)? -1.: 1.;
        d_xy[iii] = ((ii+h) & 1)? -1.: 1.;
    }
  
    return xt::eval(prediction(f, level_g, level-1, kg, hg, item) - 1./8 * d_x * (prediction(f, level_g, level-1, kg+1, hg, item) 
                                                                               - prediction(f, level_g, level-1, kg-1, hg, item))
                                                                 - 1./8 * d_y * (prediction(f, level_g, level-1, kg, hg+1, item) 
                                                                               - prediction(f, level_g, level-1, kg, hg-1, item))
                                                                 - 1./64 * d_xy * (prediction(f, level_g, level-1, kg+1, hg+1, item)
                                                                                 - prediction(f, level_g, level-1, kg+1, hg-1, item)
                                                                                 - prediction(f, level_g, level-1, kg-1, hg+1, item)
                                                                                 + prediction(f, level_g, level-1, kg-1, hg+1, item)));
}

template<class Field>
void one_time_step(Field &f)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1.;
    double s1, s2 = 1.5;
    double s3     = 1.5;

    // This gives the direction.
    double angle = M_PI / 4.0;  
    double kx = cos(angle);
    double ky = sin(angle);

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

            auto fp0 = f(0, level, k, h);
            auto f0p = f(1, level, k, h);
            auto fm0 = f(2, level, k, h);
            auto f0m = f(3, level, k, h);

            // We have to iterate over the elements on the considered boundary
            for (int l = 0; l < (1<<j); ++l)    {

                fp0 += coeff * (prediction(f, level, j,  k   *(1<<j) - 1, h*(1<<j) + l, 0)
                              - prediction(f, level, j, (k+1)*(1<<j) - 1, h*(1<<j) + l, 0));
                
                f0p += coeff * (prediction(f, level, j,  k*(1<<j) + l,  h   *(1<<j) - 1, 1)
                              - prediction(f, level, j,  k*(1<<j) + l, (h+1)*(1<<j) - 1, 1));

                fm0 += coeff * (prediction(f, level, j, (k+1)*(1<<j), h*(1<<j) + l, 2)
                              - prediction(f, level, j,  k   *(1<<j), h*(1<<j) + l, 2));
                
                f0m += coeff * (prediction(f, level, j,  k*(1<<j) + l, (h+1)*(1<<j), 3)
                              - prediction(f, level, j,  k*(1<<j) + l,  h   *(1<<j), 3));
            }

            // We compute the advected momenti
            auto m0 = xt::eval(                 fp0 + f0p + fm0 + f0m) ;
            auto m1 = xt::eval(lambda        * (fp0       - fm0      ));
            auto m2 = xt::eval(lambda        * (      f0p       - f0m));
            auto m3 = xt::eval(lambda*lambda * (fp0 - f0p + fm0 - f0m));

            m1 = (1 - s1) * m1 + s1 * (kx*m0*m0/2.0);
            m2 = (1 - s2) * m2 + s2 * (ky*m0*m0/2.0);
            m3 = (1 - s3) * m3 + s3 * (m0/2.0); // We can change this but normally it works fine

            // We come back to the distributions
            new_f(0, level, k, h) = .25 * m0 + .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
            new_f(1, level, k, h) = .25 * m0                    + .5/lambda * (m2) - .25/(lambda*lambda) * m3;
            new_f(2, level, k, h) = .25 * m0 - .5/lambda * (m1)                    + .25/(lambda*lambda) * m3;
            new_f(3, level, k, h) = .25 * m0                    - .5/lambda * (m2) - .25/(lambda*lambda) * m3;


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
    str << "LBM_D2Q4_Burgers_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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
    cxxopts::Options options("lbm_d2q4_burgers",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar Burgers equation");

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

            mure::Box<double, dim> box({-3, 3}, {3, 3});
            mure::Mesh<Config> mesh{box, min_level, max_level};
            // mure::Mesh<Config> mesh_old{box, min_level, max_level};

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
