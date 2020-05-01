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

double sq = 1.9;//1./(.5 + sigma_q);
double sxy = 1./(.5 + sigma_xy);

double kx = 0.2;
double ky = 0.5;

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

        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                <= radius * radius)
            m0 = 1;

        double m1 = kx*m0;
        double m2 = ky*m0;
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

template<class Field, class interval_t, class index_t>
auto prediction_with_mem(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const index_t j, const std::size_t item,
                         std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t, index_t>, 
                         xt::xtensor<double, 1>> & mem_map)
{
    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i, j});
    if (it != mem_map.end())   {
        //std::cout<<std::endl<<"Found by memoization";
        return it->second;
    }
    else
    {
        auto mesh = f.mesh();
        xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});
        auto mask = mesh.exists(level_g + level, i, j);

        if (xt::all(mask))
        {         
            return xt::eval(f(item, level_g + level, i, j));
        }    

        auto step = i.step;
        auto ig = i / 2;
        auto jg = j / 2;
        ig.step = step >> 1;
        xt::xtensor<double, 1> d_x = xt::empty<double>({i.size()/i.step});
        xt::xtensor<double, 1> d_xy = xt::empty<double>({i.size()/i.step});
        double d_y = (j & 1)? -1.: 1.;

        for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
        {
            d_x[iii] = (ii & 1)? -1.: 1.;
            d_xy[iii] = ((ii+j) & 1)? -1.: 1.;
        }
    
        auto val = xt::eval(prediction_with_mem(f, level_g, level-1, ig, jg, item, mem_map) - 1./8 * d_x * (prediction_with_mem(f, level_g, level-1, ig+1, jg, item, mem_map)
                                                                                                          - prediction_with_mem(f, level_g, level-1, ig-1, jg, item, mem_map))
                                                                                            - 1./8 * d_y * (prediction_with_mem(f, level_g, level-1, ig, jg+1, item, mem_map) 
                                                                                                          - prediction_with_mem(f, level_g, level-1, ig, jg-1, item, mem_map))
                                                                                          - 1./64 * d_xy * (prediction_with_mem(f, level_g, level-1, ig+1, jg+1, item, mem_map)
                                                                                                          - prediction_with_mem(f, level_g, level-1, ig+1, jg-1, item, mem_map)
                                                                                                          - prediction_with_mem(f, level_g, level-1, ig-1, jg+1, item, mem_map)
                                                                                                          + prediction_with_mem(f, level_g, level-1, ig-1, jg+1, item, mem_map)));

        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1}, j)[0];
            }
        }

        // The value should be added to the memoization map before returning
        return mem_map[{item, level_g, level, i, j}] = out;
        // return out;
    }
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

            m1 = (1 - sq) * m1 + sq * kx * m0;
            m2 = (1 - sq) * m2 + sq * ky * m0;
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
void one_time_step_with_mem(Field &f)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    mure::mr_prediction(f);

    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    using interval_t = typename Field::Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t, coord_index_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

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
            memoization_map.clear();
            for (int l = 0; l < (1<<j); ++l)
            {
                f0 += coeff * (prediction_with_mem(f, level, j,  k   *(1<<j) - 1, h*(1<<j) + l, 0, memoization_map)
                              - prediction_with_mem(f, level, j, (k+1)*(1<<j) - 1, h*(1<<j) + l, 0, memoization_map));
            }

            memoization_map.clear();
            for (int l = 0; l < (1<<j); ++l)
            {
                f1 += coeff * (prediction_with_mem(f, level, j,  k*(1<<j) + l,  h   *(1<<j) - 1, 1, memoization_map)
                              - prediction_with_mem(f, level, j,  k*(1<<j) + l, (h+1)*(1<<j) - 1, 1, memoization_map));
            }

            memoization_map.clear();
            for (int l = 0; l < (1<<j); ++l)
            {
                f2 += coeff * (prediction_with_mem(f, level, j, (k+1)*(1<<j), h*(1<<j) + l, 2, memoization_map)
                              - prediction_with_mem(f, level, j,  k   *(1<<j), h*(1<<j) + l, 2, memoization_map));
            }

            memoization_map.clear();
            for (int l = 0; l < (1<<j); ++l)
            {
                f3 += coeff * (prediction_with_mem(f, level, j,  k*(1<<j) + l, (h+1)*(1<<j), 3, memoization_map)
                              - prediction_with_mem(f, level, j,  k*(1<<j) + l,  h   *(1<<j), 3, memoization_map));
            }

            // We compute the advected momenti
            auto m0 = xt::eval(                 f0 + f1 + f2 + f3) ;
            auto m1 = xt::eval(lambda        * (f0      - f2      ));
            auto m2 = xt::eval(lambda        * (     f1      - f3));
            auto m3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));

            m1 = (1 - sq) * m1 + sq * kx * m0;
            m2 = (1 - sq) * m2 + sq * ky * m0;
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
    str << "LBM_D2Q4_advection_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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
    cxxopts::Options options("lbm_d2q4_advection",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar advection equation");

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
                    if (refinement(f, eps, 300.0, i))
                        break;
                }

                // one_time_step_with_mem(f);
                one_time_step(f,pred_coeff);
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
