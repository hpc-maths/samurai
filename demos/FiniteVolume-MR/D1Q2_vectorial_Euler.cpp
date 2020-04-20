#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <xtensor/xio.hpp>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"

#include <chrono>


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


std::array<double, 3> solution(double x, double t)   {

    double density  = (x <= 0.0) ? 1.0 : 0.125;
    double velocity =              0.0;
    double pressure = (x <= 0.0) ? 1.0 : 0.1;

    return {density, velocity, pressure};
}




template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 6;
    mure::BC<1> bc{ {{ {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0},
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    double gamma = 1.4;

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];


        auto initial_data = solution(x, 0.0);


        double density  = initial_data[0];
        double velocity = initial_data[1];
        double pressure = initial_data[2];

        double u10 = density;
        double u20 = density * velocity;
        double u30 = 0.5 * density * velocity * velocity + pressure / (gamma - 1.0);

        double u11 = u20;
        double u21 = (gamma - 1.0) * u30 + (3.0 - gamma)/(2.0) * (u20*u20)/u10; 
        double u31 = gamma * (u20*u30)/(u10) + (1.0 - gamma)/2.0 * (u20*u20*u20)/(u10*u10);


        f[cell][0] = .5 * (u10 + u11);
        f[cell][1] = .5 * (u10 - u11);

        f[cell][2] = .5 * (u20 + u21);
        f[cell][3] = .5 * (u20 - u21);

        f[cell][4] = .5 * (u30 + u31);
        f[cell][5] = .5 * (u30 - u31);
    });

    return f;
}

template<class Field, class interval_t, class FieldTag>
xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item, 
                                  const FieldTag & tag, std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 1>> & mem_map)
{

    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i});
    if (it != mem_map.end())   {
        //std::cout<<std::endl<<"Found by memoization";
        return it->second;
    }
    else {

        auto mesh = f.mesh();
        xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});//xt::eval(f(item, level_g, i));
        auto mask = mesh.exists(level_g + level, i, mure::MeshType::cells_and_ghosts);

        // std::cout << level_g + level << " " << i << " " << mask << "\n"; 
        if (xt::all(mask))
        {         
            return xt::eval(f(item, level_g + level, i));
        }

        auto step = i.step;
        auto ig = i / 2;
        ig.step = step >> 1;
        xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

        for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
        {
            d[iii] = (ii & 1)? -1.: 1.;
        }

    
        auto val = xt::eval(prediction(f, level_g, level-1, ig, item, tag, mem_map) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, tag, mem_map) 
                                                                                       - prediction(f, level_g, level-1, ig-1, item, tag, mem_map)));
        

        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
            }
        }

        // The value should be added to the memoization map before returning
        return mem_map[{item, level_g, level, i}] = out;

        //return out;
    }

}


template<class Field, class FieldTag>
void one_time_step(Field &f, const FieldTag & tag, double s)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 3.;//, s = 1.0;
    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    mure::mr_prediction(f);


    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    using interval_t = typename Field::Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];



            std::size_t j = max_level - level;

            double coeff = 1. / (1 << j);

            std::cout<<std::endl<<"Level "<<level<<" Interval "<<i<<" At finest = "<<i*(1<<j)<<" Minus one "<<i*(1<<j)-1<<" Plus one "<<(i+1)*(1<<j)<<std::endl;
            auto fp1 = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, tag, memoization_map));
            std::cout<<"Plus"<<std::endl;
            auto fm1 = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, tag, memoization_map));
            std::cout<<"Minus"<<std::endl;

            auto fp2 = f(2, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 2, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 2, tag, memoization_map));
            auto fm2 = f(3, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 3, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 3, tag, memoization_map));

            auto fp3 = f(4, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 4, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 4, tag, memoization_map));
            auto fm3 = f(5, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 5, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 5, tag, memoization_map));

            std::cout<<std::endl<<"Done";

           
            auto u10 = xt::eval(          fp1 + fm1);
            auto u11 = xt::eval(lambda * (fp1 - fm1));
           
            auto u20 = xt::eval(          fp2 + fm2);
            auto u21 = xt::eval(lambda * (fp2 - fm2));
           
            auto u30 = xt::eval(          fp3 + fm3);
            auto u31 = xt::eval(lambda * (fp3 - fm3));


            double gamma = 1.4;
            auto u11_coll = (1 - s) * u11 + s * (u20);

            auto u21_coll = (1 - s) * u21 + s * ((gamma - 1.0) * u30 + (3.0 - gamma)/(2.0) * (u20*u20)/u10);

            auto u31_coll = (1 - s) * u31 + s * (gamma * (u20*u30)/(u10) + (1.0 - gamma)/2.0 * (u20*u20*u20)/(u10*u10));


            new_f(0, level, i) = .5 * (u10 + 1. / lambda * u11_coll);
            new_f(1, level, i) = .5 * (u10 - 1. / lambda * u11_coll);

            new_f(2, level, i) = .5 * (u20 + 1. / lambda * u21_coll);
            new_f(3, level, i) = .5 * (u20 - 1. / lambda * u21_coll);

            new_f(4, level, i) = .5 * (u30 + 1. / lambda * u31_coll);
            new_f(5, level, i) = .5 * (u30 - 1. / lambda * u31_coll);
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
    str << "LBM_D1Q2_Vectorial_Euler_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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
    cxxopts::Options options("lbm_d1q2 vectorial for euler",
                             "...");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("9"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.001"))
                       ("s", "relaxation parameter", cxxopts::value<double>()->default_value("1.0"))
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
            using Config = mure::MRConfig<dim, 6>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();
            double s = result["s"].as<double>();


            mure::Box<double, dim> box({-1}, {1});
            mure::Mesh<Config> mesh{box, min_level, max_level};

            // Initialization
            auto f  = init_f(mesh , 0.0);       

            double T = 0.4;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / 3.0;

            std::size_t N = static_cast<std::size_t>(T / dt);

            double t = 0.0;



            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {

                std::cout<<std::endl<<"Iteration "<<nb_ite<<" time = "<<t;
                tic();
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (coarsening(f, eps, i))
                        break;
                }
                auto duration_coarsening = toc();

                // save_solution(f, eps, nb_ite, "coarsening");

                tic();
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (refinement(f, eps, i))
                        break;
                }
                auto duration_refinement = toc();
                //save_solution(f, eps, nb_ite, "refinement");


                // Create and initialize field containing the leaves
                tic();
                mure::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                tag_leaf.array().fill(0);
                mesh.for_each_cell([&](auto &cell) {
                    tag_leaf[cell] = static_cast<int>(1);
                });
                auto duration_leaf_checking = toc();


                std::cout<<std::endl;

                

                
                tic();
                one_time_step(f, tag_leaf, s);
                auto duration_scheme = toc();


                t += dt;

                tic();
                save_solution(f, eps, nb_ite, "onetimestep");
                auto duration_save = toc();
                                    

            }
            

        }

    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
