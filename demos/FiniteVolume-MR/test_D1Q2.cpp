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


// To decide which test we want to perform.

/*

1 : transport - gaussienne
2 : transport - probleme de Riemann
3 : Burgers - tangente hyperbolique reguliere
4 : Burgers - fonction chapeau avec changement de regularite
5 : Burgers - probleme de Riemann
*/

const int test_number = 5;

const double ad_vel = 0.75; // Should be < lambda




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

double exact_solution(double x, double t)   {

    double u = 0;


    switch(test_number) {
        case 1 : {
            u = exp(-20.0 * (x-ad_vel*t) * (x-ad_vel*t));
            break;
        }
        
        case 2 : {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            double xtr = x - ad_vel*t;
            u =  (xtr <= -sigma) ? (rhoL) : ((xtr <= sigma) ? (rhoC) : rhoR );
            break;
        }
        case 3 : {
            double sigma = 100.0;
            if (t <= 0.0)
                u = 0.5 * (1.0 + tanh(sigma * x));
            else
            {   // We proceed by dicothomy
                double a = -3.2;
                double b =  3.2;

                double tol = 1.0e-8;

                auto F = [sigma, x, t] (double y)   {
                    return y + 0.5 * (1.0 + tanh(sigma * y))*t - x;
                };
                double res = 0.0;

                while (b-a > tol)   {
                    double mean = 0.5 * (b + a);
                    double eval = F(mean);
                    if (eval <= 0.0)
                        a = mean;
                    else
                        b = mean;
                    res = mean;
                }
            
                u =  0.5 * (1.0 + tanh(sigma * res));
            }
            break;
        }

        case 4 : {
            if (x >= -1 and x < t)
            {
                u = (1 + x) / (1 + t);
            }

            if (x >= t and x < 1)
            {
                u = (1 - x) / (1 - t);
            }
            break;
        }

        case 5 : {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            u =  (x + sigma <= rhoL * t) ? rhoL : ((x + sigma <= rhoC*t) ? (x+sigma)/t : ((x-sigma <= t/2*(rhoC + rhoR)) ? rhoC : rhoR ));
            break;
        }
    }
    
    return u;
}

double flux(double u)   {

    if (test_number == 1 or test_number == 2)   {
        return ad_vel * u;
    }
    else
    {
        return 0.5 * u *u;
    }
    

}

template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 2;
    mure::BC<1> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        double u = 0;

        u = exact_solution(x, 0.0);


        double v = flux(u);

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
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
        auto mask = mesh.exists(level_g + level, i);

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


template<class Field, class interval_t>
xt::xtensor<double, 2> prediction_all(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, 
                                  std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 2>> & mem_map)
{

    using namespace xt::placeholders;
    // We check if the element is already in the map
    auto it = mem_map.find({level_g, level, i});
    if (it != mem_map.end())
    {
        return it->second;
    }
    else
    {
        auto mesh = f.mesh();
        std::vector<std::size_t> shape = {i.size(), 2};
        xt::xtensor<double, 2> out = xt::empty<double>(shape);
        auto mask = mesh.exists(level_g + level, i);

        xt::xtensor<double, 2> mask_all = xt::empty<double>(shape);
        xt::view(mask_all, xt::all(), 0) = mask;
        xt::view(mask_all, xt::all(), 1) = mask;

        if (xt::all(mask))
        {         
            return xt::eval(f(level_g + level, i));
        }

        auto ig = i >> 1;
        ig.step = 1;

        xt::xtensor<double, 2> val = xt::empty<double>(shape);
        auto current = xt::eval(prediction_all(f, level_g, level-1, ig, mem_map));
        auto left = xt::eval(prediction_all(f, level_g, level-1, ig-1, mem_map));
        auto right = xt::eval(prediction_all(f, level_g, level-1, ig+1, mem_map));

        std::size_t start_even = (i.start&1)? 1: 0;
        std::size_t start_odd = (i.start&1)? 0: 1;
        std::size_t end_even = (i.end&1)? ig.size(): ig.size()-1;
        std::size_t end_odd = (i.end&1)? ig.size()-1: ig.size();
        xt::view(val, xt::range(start_even, _, 2)) = xt::view(current - 1./8 * (right - left), xt::range(start_even, _));
        xt::view(val, xt::range(start_odd, _, 2)) = xt::view(current + 1./8 * (right - left), xt::range(_, end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, ++i_int)
        {
            if (mask[i_mask])
            {
                xt::view(out, i_mask) = xt::view(f(level_g + level, {i_int, i_int + 1}), 0);
            }
        }

        // The value should be added to the memoization map before returning
        return out;// mem_map[{level_g, level, i, ig}] = out;
    }
}

template<class Field, class FieldTag>
void one_time_step(Field &f, const FieldTag & tag, double s)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1.;//, s = 1.0;
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


            // STREAM

            std::size_t j = max_level - level;

            double coeff = 1. / (1 << j);

            // This is the STANDARD FLUX EVALUATION
            
            auto fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, tag, memoization_map));

            auto fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, tag, memoization_map));
            
    

            // COLLISION    

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            if (test_number == 1 or test_number == 2)   {
                vv = (1 - s) * vv + s * ad_vel * uu;
            }
            else
            {
                vv = (1 - s) * vv + s * .5 * uu * uu;
            }
            
            new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}

// template<class Field, class FieldR>
template<class Config, class FieldR>
std::array<double, 2> compute_error(mure::Field<Config, double, 2> &f, FieldR & fR, double t)
{

    auto mesh = f.mesh();

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    mure::mr_projection(f);
    mure::mr_prediction(f);  // C'est supercrucial de le faire.

    f.update_bc(); // Important especially when we enforce Neumann...for the Riemann problem
    fR.update_bc();    

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;
    error_memoization_map.clear();

    double error = 0; // To return
    double diff = 0.0;


    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(meshR[mure::MeshType::cells][max_level],
                                      mesh[mure::MeshType::cells][level])
                  .on(max_level);

        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));


            xt::xtensor<double, 1> x = dx*xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5*dx;
            // xt::xtensor<double, 1> uexact = (x >= -1.0 and x < t) * ((1 + x) / (1 + t)) + 
            //                                 (x >= t and x < 1) * (1 - x) / (1 - t);

            xt::xtensor<double, 1> uexact = xt::zeros<double>(x.shape());

            for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)    {
                uexact[idx] = exact_solution(x[idx], t); // We can probably do better
            }

            error += xt::sum(xt::abs(xt::flatten(xt::view(fR(max_level, i), xt::all(), xt::range(0, 1)) + xt::view(fR(max_level, i), xt::all(), xt::range(1, 2))) 
                             - uexact))[0];


            diff += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), xt::range(0, 1)) + xt::view(sol, xt::all(), xt::range(1, 2))) - xt::flatten(xt::view(fR(max_level, i), xt::all(), xt::range(0, 1)) + xt::view(fR(max_level, i), xt::all(), xt::range(1, 2)))))[0];


        });
    }

    return {dx * error, dx * diff}; // Normalization by dx before returning
    // I think it is better to do the normalization at the very end ... especially for round-offs    
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers",
                             "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("10"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))
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
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = 2;//result["min_level"].as<std::size_t>();
            std::size_t max_level = 9;//result["max_level"].as<std::size_t>();


            // We set some parameters according
            // to the problem.
            double sol_reg = 0.0;
            double T = 0.0;
            std::string case_name;

            switch(test_number){
                case 1 : {
                    sol_reg = 600.0; // The solution is very smooth
                    T = 0.4;
                    case_name = std::string("t_r");
                    break;
                }
                case 2 : {
                    sol_reg = 0.0;
                    T = 0.4;
                    case_name = std::string("t_d");
                    break;
                }
                case 3 : {
                    sol_reg = 600.0;
                    T = 0.4;
                    case_name = std::string("b_r");
                    break;
                }
                case 4 : {
                    sol_reg = 0.0;
                    T = 1.3; // Let it develop the discontnuity
                    case_name = std::string("b_c");
                    break;
                }
                case 5 : {
                    sol_reg = 0.0;
                    T = 0.7;
                    case_name = std::string("b_d");
                    break;
                }
            }

            mure::Box<double, dim> box({-3}, {3});

            std::vector<double> s_vect {0.75, 1.0, 1.25, 1.5, 1.75};

            for (auto s : s_vect)   {
                std::cout<<std::endl<<"Relaxation parameter s = "<<s;

                std::string prefix (case_name + "_s_"+std::to_string(s)+"_");

                std::cout<<std::endl<<"Testing time behavior"<<std::endl;
                {
                    double eps = 1.0e-4; // This remains fixed

                    mure::Mesh<Config> mesh{box, min_level, max_level};
                    mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                    // Initialization
                    auto f  = init_f(mesh , 0.0);
                    auto fR = init_f(meshR, 0.0);             

                    double dx = 1.0 / (1 << max_level);
                    double dt = dx; // Since lb = 1

                    std::size_t N = static_cast<std::size_t>(T / dt);

                    double t = 0.0;

                    std::ofstream out_time_frames;
                    std::ofstream out_error_exact_ref;
                    std::ofstream out_diff_ref_adap;
                    std::ofstream out_compression;

                    out_time_frames.open     ("./d1q2/time/"+prefix+"time.dat");
                    out_error_exact_ref.open ("./d1q2/time/"+prefix+"error.dat");
                    out_diff_ref_adap.open   ("./d1q2/time/"+prefix+"diff.dat");
                    out_compression.open     ("./d1q2/time/"+prefix+"comp.dat");


                    for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                    {
                        for (std::size_t i=0; i<max_level-min_level; ++i)
                        {
                            if (coarsening(f, eps, i))
                                break;
                        }

                        for (std::size_t i=0; i<max_level-min_level; ++i)
                        {
                            if (refinement(f, eps, sol_reg, i))
                                break;
                        }

                        mure::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                        tag_leaf.array().fill(0);
                        mesh.for_each_cell([&](auto &cell) {
                            tag_leaf[cell] = static_cast<int>(1);
                        });
        
                        mure::Field<Config, int, 1> tag_leafR{"tag_leafR", meshR};
                        tag_leafR.array().fill(0);
                        meshR.for_each_cell([&](auto &cell) {
                            tag_leafR[cell] = static_cast<int>(1);
                        });

                        auto error = compute_error(f, fR, t);

                        out_time_frames    <<t       <<std::endl;
                        out_error_exact_ref<<error[0]<<std::endl;
                        out_diff_ref_adap  <<error[1]<<std::endl;
                        out_compression    <<static_cast<double>(mesh.nb_cells(mure::MeshType::cells)) 
                                           / static_cast<double>(meshR.nb_cells(mure::MeshType::cells))<<std::endl;

                        std::cout<<std::endl<<"Time = "<<t<<" Diff = "<<error[1];

                
                        one_time_step(f, tag_leaf, s);
                        one_time_step(fR, tag_leafR, s);

                        t += dt;
             
                    }

                    std::cout<<std::endl;
            
                    out_time_frames.close();
                    out_error_exact_ref.close();
                    out_diff_ref_adap.close();
                    out_compression.close();
                }
                
                std::cout<<std::endl<<"Testing eps behavior"<<std::endl;
                {
                    double eps = 0.1;
                    std::size_t N_test = 50;
                    double factor = 0.60;
                    std::ofstream out_eps;
                    std::ofstream out_diff_ref_adap;
                    std::ofstream out_compression;

                    out_eps.open             ("./d1q2/eps/"+prefix+"eps.dat");
                    out_diff_ref_adap.open   ("./d1q2/eps/"+prefix+"diff.dat");
                    out_compression.open     ("./d1q2/eps/"+prefix+"comp.dat");

                    for (std::size_t n_test = 0; n_test < N_test; ++ n_test)    {
                        std::cout<<std::endl<<"Test "<<n_test<<" eps = "<<eps;

                        mure::Mesh<Config> mesh{box, min_level, max_level};
                        mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                        // Initialization
                        auto f  = init_f(mesh , 0.0);
                        auto fR = init_f(meshR, 0.0);             

                        double dx = 1.0 / (1 << max_level);
                        double dt = dx; // Since lb = 1

                        std::size_t N = static_cast<std::size_t>(T / dt);

                        double t = 0.0;


                        double foo_diff = 0.0;
                        
                        for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        {
                        
                            for (std::size_t i=0; i<max_level-min_level; ++i)
                            {
                                if (coarsening(f, eps, i))
                                    break;
                            }

                            for (std::size_t i=0; i<max_level-min_level; ++i)
                            {
                                if (refinement(f, eps, sol_reg, i))
                                    break;
                            }
            

                            mure::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                            tag_leaf.array().fill(0);
                            mesh.for_each_cell([&](auto &cell) {
                                tag_leaf[cell] = static_cast<int>(1);
                            });

                            mure::Field<Config, int, 1> tag_leafR{"tag_leafR", meshR};
                            tag_leafR.array().fill(0);
                            meshR.for_each_cell([&](auto &cell) {
                                tag_leafR[cell] = static_cast<int>(1);
                            });





      

                            one_time_step(f, tag_leaf, s);
                            one_time_step(fR, tag_leafR, s);


                            { // This is ultra important if we do not want to compute the error
                            // at each time step.
                                mure::mr_projection(f);
                                mure::mr_prediction(f); 

                                f.update_bc(); //
                                fR.update_bc();    
                            }


                            t += dt;
         
                        }


                        auto error = compute_error(f, fR, 0.0);
                        std::cout<<"Diff = "<<error[1]<<std::endl;
                            
                            
                        
                        out_eps<<eps<<std::endl;
                        out_diff_ref_adap<<error[1]<<std::endl;
                        out_compression<<static_cast<double>(mesh.nb_cells(mure::MeshType::cells)) 
                                           / static_cast<double>(meshR.nb_cells(mure::MeshType::cells))<<std::endl;

                        eps *= factor;
                    }
            
                    out_eps.close();            
                    out_diff_ref_adap.close();
                    out_compression.close();

                }
            }
        }
    }
    
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
