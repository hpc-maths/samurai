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

#include "prediction_map_1d.hpp"



template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(2);

        data[k][0] = prediction(k, i*size - 1) - prediction(k, (i+1)*size - 1);
        data[k][1] = prediction(k, (i+1)*size) - prediction(k, i*size);
    }
    return data;
}


// std::array<double, 4> mult_by_prediction_matrix(std::size_t exponent, const std::array<double, 4> & in)
// {
//     std::array<double, 4> out;
//     out.fill(0.0);

//     std::vector<std::array<std::array<double, 4>, 4>> matrices;

//     // exp 1
//     std::array<std::array<double, 4>, 4> tmp {std::array<double, 4>{1./8., 1., -1./8., 0.},
//                                               std::array<double, 4>{-1./8., 1., 1./8., 0.}, 
//                                               std::array<double, 4>{0., 1./8., 1., -1./8.}, 
//                                               std::array<double, 4>{0., -1./8., 1., 1./8.}};
//     matrices.push_back(tmp);   

//     // exp 2
//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-7./64, 71./64, -1./64, 1./64},
//                                                std::array<double, 4>{-9./64, 57./64, 17./64, -1./64}, 
//                                                std::array<double, 4>{-1./64, 17./64, 57./64, -9./64}, 
//                                                std::array<double, 4>{1./64, -1./64, 71./64, -7./64}};
//     matrices.push_back(tmp);  

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-39./256, 255./256, 39./256, 1./256}, 
//                                                std::array<double, 4>{-33./256, 201./256, 97./256, -9./256},  
//                                                std::array<double, 4>{-9./256, 97./256, 201./256, -33./256},  
//                                                std::array<double, 4>{ 1./256, 39./256, 255./256, -39./256}};
//     matrices.push_back(tmp);   

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-147./1024, 883./1024, 307./1024, -19./1024}, 
//                                                std::array<double, 4>{-117./1024, 725./1024, 469./1024, -53./1024},   
//                                                std::array<double, 4>{-53./1024, 469./1024, 725./1024, -117./1024},   
//                                                std::array<double, 4>{-19./1024, 307./1024, 883./1024, -147./1024}};
//     matrices.push_back(tmp);   

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-515./4096, 3107./4096, 1667./4096, -163./4096}, 
//                                                std::array<double, 4>{-421./4096, 2693./4096, 2085./4096, -261./4096},   
//                                                std::array<double, 4>{-261./4096, 2085./4096, 2693./4096, -421./4096},   
//                                                std::array<double, 4>{-163./4096, 1667./4096, 3107./4096, -515./4096}};
//     matrices.push_back(tmp);

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-1811./16384, 11283./16384, 7827./16384, -915./16384}, 
//                                                std::array<double, 4>{-1557./16384, 10261./16384, 8853./16384, -1173./16384},   
//                                                std::array<double, 4>{-1173./16384, 8853./16384, 10261./16384, -1557./16384},   
//                                                std::array<double, 4>{-915./16384, 7827./16384, 11283./16384, -1811./16384}};
//     matrices.push_back(tmp); 

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-6547./65536, 42259./65536, 34195./65536, -4371./65536}, 
//                                                std::array<double, 4>{-5909./65536, 39829./65536, 36629./65536, -5013./65536},  
//                                                std::array<double, 4>{-5013./65536, 36629./65536, 39829./65536, -5909./65536},  
//                                                std::array<double, 4>{-4371./65536, 34195./65536, 42259./65536, -6547./65536}};
//     matrices.push_back(tmp); 

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-24403./262144, 162131./262144, 143699./262144, -19283./262144},  
//                                                std::array<double, 4>{-22869./262144, 156501./262144, 149333./262144, -20821./262144},   
//                                                std::array<double, 4>{-20821./262144, 149333./262144, 156501./262144, -22869./262144},   
//                                                std::array<double, 4>{-19283./262144, 143699./262144, 162131./262144, -24403./262144}};
//     matrices.push_back(tmp); 

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-93267./1048576, 632403./1048576, 590931./1048576, -81491./1048576},   
//                                                std::array<double, 4>{-89685./1048576, 619605./1048576, 603733./1048576, -85077./1048576},    
//                                                std::array<double, 4>{-85077./1048576, 603733./1048576, 619605./1048576, -89685./1048576},    
//                                                std::array<double, 4>{-81491./1048576, 590931./1048576, 632403./1048576, -93267./1048576}};
//     matrices.push_back(tmp); 

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-362835./4194304, 2492755./4194304, 2400595./4194304, -336211./4194304},    
//                                                std::array<double, 4>{-354645./4194304, 2464085./4194304, 2429269./4194304, -344405./4194304},     
//                                                std::array<double, 4>{-344405./4194304, 2429269./4194304, 2464085./4194304, -354645./4194304},     
//                                                std::array<double, 4>{-336211./4194304, 2400595./4194304, 2492755./4194304, -362835./4194304}};
//     matrices.push_back(tmp); 

//     tmp = std::array<std::array<double, 4>, 4>{std::array<double, 4>{-1427795./16777216, 9888083./16777216, 9685331./16777216, -1368403./16777216},     
//                                                std::array<double, 4>{-1409365./16777216, 9824597./16777216, 9748821./16777216, -1386837./16777216},      
//                                                std::array<double, 4>{-1386837./16777216, 9748821./16777216, 9824597./16777216, -1409365./16777216},      
//                                                std::array<double, 4>{-1368403./16777216, 9685331./16777216, 9888083./16777216, -1427795./16777216}};
//     matrices.push_back(tmp); 

//     std::size_t which_matrix = exponent - 1;

//     for (std::size_t i = 0; i < 4; ++i) {
//         out[i] = 0.0;
//         for (std::size_t j = 0; j < 4; ++j) {
//             out[i] += matrices[which_matrix][i][j] * in[j];
//         }
//     }

//     return out;
// }


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

    // { // Hyperbolic tangent

    //     double sigma = 20.0;

    //     if (t <= 0.0)
    //         return 0.5 * (1.0 + tanh(sigma * x));
    //     else
    //     {   // We proceed by dicothomy
    //         double a = -3.2;
    //         double b =  3.2;

    //         double tol = 1.0e-8;

    //         auto F = [sigma, x, t] (double y)   {
    //             return y + 0.5 * (1.0 + tanh(sigma * y))*t - x;
    //         };
    //         double res = 0.0;

    //         while (b-a > tol)   {
    //             double mean = 0.5 * (b + a);
    //             double eval = F(mean);
    //             if (eval <= 0.0)
    //                 a = mean;
    //             else
    //                 b = mean;
    //             res = mean;
    //         }

    //         return 0.5 * (1.0 + tanh(sigma * res));

    //     }
        
    // }

    // double rhoL = 1.0;
    // double rhoR = 0.0;
    // double x0 = 0.0;

    // double vshock = 0.5 * (rhoL + rhoR);

    // //return ((x-0.5*t) <= x0) ? rhoL : rhoR;
    // //return ((x-0.5*t) <= -1.0) ? 0 : ((x-0.5*t) <= 1.0 ? 1.0 : 0.0);

    // return (x <= x0 + vshock * t) ? rhoL : rhoR;
    // {
    //     double x0L = -0.2;
    //     double x0R =  0.2;
    //     return ((x - 0.5*t) < x0L) ? 0.0 : (((x - 0.5*t) < x0R ? 1.0 : 0.0));
    // }

    double sigma = 0.5;
    double rhoL = 0.0;
    double rhoC = 1.0;
    double rhoR = 0.0;

    // We translate up just to see if there are 0s left

    //return (x + sigma <= rhoL * t) ? rhoL : ((x + sigma <= rhoC*t) ? (x+sigma)/t : ((x-sigma <= t/2*(rhoC + rhoR)) ? rhoC : rhoR ));
    if (x < -sigma){
        return 0.;

    }
    else
    {
        if (x < sigma){
            return 1.;

        }
        else 
            return 0.;
    }
    

    // // x = x - 0.5 * t;
    // // t = 0.0;

    // if (x >= -1 and x < t)
    // {
    //     u = (1 + x) / (1 + t);
    // }
    
    // if (x >= t and x < 1)
    // {
    //     u = (1 - x) / (1 - t);
    // }

    // u = exp(-20.0 * (x-0.75*t) * (x-0.75*t));

    // u = 0.0;
    
    return u;
}

double flux(double u)   {
    //return 0.5 * u * u;
    return 0.75 * u;
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

        // if (x >= -1 and x < t)
        // {
        //     u = (1 + x) / (1 + t);
        // }
        // if (x >= t and x < 1)
        // {
        //     u = (1 - x) / (1 - t);
        // }

        u = exact_solution(x, 0.0);

        //double u = exp(-20.0 * x * x);

        double v = flux(u);//.5 * u; 
        //double v = .5 * u * u;

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
    });

    return f;
}

template<class Field, class interval_t>
xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item, 
                                  std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 1>> & mem_map, bool cheap = false)
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
        auto mask = mesh.exists(mure::MeshType::cells_and_ghosts, level_g + level, i);

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

        
        xt::xtensor<double, 1> val;

        if (cheap)  {  // This is the cheap prediction
            val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap));
        }
        else {
            val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, mem_map, cheap) 
                                                                                       - prediction(f, level_g, level-1, ig-1, item, mem_map, cheap)));
        }
    

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
        auto mask = mesh.exists(mure::MeshType::cells_and_ghosts, level_g + level, i);

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

// template<class Field>
// double prediction_matrix(const Field& f, volatile int k, volatile std::size_t n_field, volatile std::size_t calling_level)
// {
//     using interval_t = typename Field::Config::interval_t;
//     volatile std::size_t copy_calling_level = calling_level;

//     std::cout<<std::endl<<"Cell k = "<<k<<std::flush;

//     auto mesh = f.mesh();
//     auto max_level = mesh.max_level();
//     auto min_level = mesh.min_level();

//     if (mesh.exists(mure::MeshType::cells, max_level, interval_t(k, k+1))[0])  {
//         return xt::eval(f(n_field, max_level, interval_t(k, k+1)))[0];
//     }
//     else
//     {
//         volatile int k_finest = (k%2 == 0) ? k : k + 1; // This is what we call "k" in the theory of the prediction with the matrix
//         std::size_t level_stop = max_level;

//         bool stop = false;
//         while(!stop and level_stop >= min_level) {

//             stop = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-2, k_finest-1))[0]    or
//                    mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-1, k_finest))[0]      or
//                    mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest, k_finest+1))[0]      or
//                    mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest+1, k_finest+2))[0];

//             if (!stop)  {
//                 k_finest = k_finest / 2;
//                 level_stop--;
//             }
//         }

//         std::cout<<std::endl<<"Level stop = "<<level_stop<<std::flush<<std::endl;

//         std::array<bool, 4> is_at_stop_level;
//         // is_at_stop_level.fill(false);
//         // is_at_stop_level[0] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-2, k_finest-1))[0];
//         // is_at_stop_level[1] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-1, k_finest))[0];
//         // is_at_stop_level[2] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest, k_finest+1))[0];
//         // is_at_stop_level[3] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest+1, k_finest+2))[0];


//         is_at_stop_level[0] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-2, k_finest-1))[0] or 
//                               mesh.exists(mure::MeshType::proj_cells, level_stop, interval_t(k_finest-2, k_finest-1))[0];

//         is_at_stop_level[1] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest-1, k_finest))[0] or 
//                               mesh.exists(mure::MeshType::proj_cells, level_stop, interval_t(k_finest-1, k_finest))[0];

//         is_at_stop_level[2] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest, k_finest+1))[0] or 
//                               mesh.exists(mure::MeshType::proj_cells, level_stop, interval_t(k_finest, k_finest+1))[0];

//         is_at_stop_level[3] = mesh.exists(mure::MeshType::cells, level_stop, interval_t(k_finest+1, k_finest+2))[0] or 
//                               mesh.exists(mure::MeshType::proj_cells, level_stop, interval_t(k_finest+1, k_finest+2))[0];;

//         std::cout<<interval_t(k_finest-2, k_finest-1)<<" | "
//                  <<interval_t(k_finest-1, k_finest)<<" | "
//                  <<interval_t(k_finest, k_finest+1)<<" | "
//                  <<interval_t(k_finest+1, k_finest+2)<<" | "<<std::endl;

//         for (auto el : is_at_stop_level)
//             std::cout<<"| "<<el;
        
//         std::cout<<std::flush;

//         std::array<double, 4> in_values;
//         in_values.fill(0.0);

//         for (std::size_t idx = 0; idx < 4; ++idx)   {

//             int cell_start = k_finest - 2 + idx;

//             if (is_at_stop_level[idx])  {
//                 in_values[idx] = xt::eval(f(n_field, level_stop, interval_t(cell_start, cell_start + 1)))[0];
//             }
//             else
//             {
                
//                 in_values[idx] = xt::eval(f(n_field, level_stop - 1, interval_t(cell_start / 2, cell_start/2  + 1)))[0]
//                         + 1./8. * (cell_start % 2 == 0 ? 1. : -1. ) * (xt::eval(f(n_field, level_stop - 1, interval_t(cell_start / 2 - 1, cell_start/2)))[0]
//                                                                       -xt::eval(f(n_field, level_stop - 1, interval_t(cell_start / 2 + 1, cell_start/2 + 2)))[0]);                     
                
  
//             }            
//         }

//         auto out_values = mult_by_prediction_matrix(max_level - level_stop, in_values);

//         // Think about it more precisely
//         return (k % 2 == 0) ? out_values[2] : out_values[1];
//     }
    
// }


// template<class Field>
// void one_time_step_matrix(Field &f, double s)
// {
//     constexpr std::size_t nvel = Field::size;
//     double lambda = 1.;//, s = 1.0;
//     auto mesh = f.mesh();
//     auto max_level = mesh.max_level();
//     auto min_level = mesh.min_level();

//     mure::mr_projection(f);
//     f.update_bc();
//     mure::mr_prediction(f);

//     using interval_t = typename Field::Config::interval_t;


//     Field new_f{"new_f", mesh};
//     new_f.array().fill(0.);


//     for (std::size_t level = 0; level <= max_level; ++level)
//     {
//         auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
//                                       mesh[mure::MeshType::cells][level]);


//         std::size_t j_dist = max_level - level;

//         double coeff = 1. / (1 << j_dist);

//         exp([&](auto, auto &interval, auto) {
//             auto i = interval[0];

//             auto number_of_cells_in_interval = i.size();

//             std::cout<<std::endl<<"Level = "<<level<<" Interval = "<<i<<std::flush;



//             for (int k = i.start; k < i.end; ++k)   {


//                 std::cout<<std::endl<<std::endl<<"* Leaf k = "<<k<<std::endl;
                


//                 double fp = xt::eval(f(0, level, interval_t(k, k  + 1)))[0] + coeff * (prediction_matrix(f, k * (1<<j_dist) - 1, 0, level)
//                                                                                      - prediction_matrix(f, (k+1) * (1<<j_dist) - 1, 0, level));

//                 double fm = xt::eval(f(1, level, interval_t(k, k  + 1)))[0] + coeff * (prediction_matrix(f, (k+1) * (1<<j_dist), 1, level)
//                                                                                      - prediction_matrix(f, k * (1<<j_dist), 1, level));

//                 double rho = fp + fm;
//                 double q = lambda * (fp - fm);


//                 double q_coll = (1. - s) * q + s * 0.5 * rho*rho;

//                 fp = 0.5 * rho + 0.5 / lambda * q_coll;
//                 fm = 0.5 * rho - 0.5 / lambda * q_coll;



//                 new_f(0, level, interval_t{k, k+1}) = fp;
//                 new_f(1, level, interval_t{k, k+1}) = fm;

//             }

//         });
//     }

//     std::swap(f.array(), new_f.array());
// }



template<class Field, class Pred>
void one_time_step_matrix(Field &f, const Pred& pred_coeff, double s_rel)
{

    double lambda = 1.;

    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        bool something_at_this_level = false;

        std::size_t j = max_level - level; 
        double coeff = 1. / (1 << j);

        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);



        // // Showing problems
        // for(auto &c: pred_coeff[j][1].coeff) {
        //     auto stencil = c.first;

        //     xt::xtensor_fixed<int, xt::xshape<1>> stencil_vec;
        //     stencil_vec = {{stencil}};

        //     xt::xtensor_fixed<int, xt::xshape<1>> stencil_back_vec;
        //     stencil_back_vec = {{-stencil}};



        //     // Je ne sais pas pour quelle raison ca ne marche pas
        //     // Aucun des deux
        //     //auto problem = mure::translate(mure::intersection(mesh[mure::MeshType::cells][level], mesh[mure::MeshType::cells][level]), stencil_vec);
        //     auto problem = mure::translate(mesh[mure::MeshType::cells][level], stencil_vec);

        //     problem([&](auto, auto &interval, auto) {
        //         auto k = interval[0]; 

        //         std::cout<<std::endl<<"Level "<<level<<" Stencil "<<stencil<<" Problem "<<k<<std::flush;

        //     });

        // }






        exp([&](auto, auto &interval, auto) {
            something_at_this_level = true;

            auto k = interval[0]; // Logical index in x


            auto fp = xt::eval(f(0, level, k));
            auto fm = xt::eval(f(1, level, k));

            //std::cout<<std::endl<<"Level = "<<level<<" Interval = "<<k<<std::endl<<std::flush;

 
            for(auto &c: pred_coeff[j][0].coeff)
            {
                coord_index_t stencil = c.first;
                double weight = c.second;

                //std::cout<<"stencil = "<<stencil<<" weight = "<<weight<<std::endl;

                fp += coeff * weight * f(0, level, k + stencil);
            }

            for(auto &c: pred_coeff[j][1].coeff)
            {
                coord_index_t stencil = c.first;
                double weight = c.second;

                fm += coeff * weight * f(1, level, k + stencil);
            }

            // COLLISION    

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));
            
            //vv = (1 - s_rel) * vv + s_rel * 0.75 * uu;
            vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

            new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}


template<class Field, class Pred>
void one_time_step_matrix_overleaves(Field &f, const Pred& pred_coeff, double s_rel)
{

    double lambda = 1.;

    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);

    // After that everything is ready, we predict what is remaining
    mure::mr_prediction_overleaves(f);


    // We check the values on the overleaves
    // for (std::size_t level = 0; level <= max_level; ++level) {

    //     auto overleaves = mure::intersection(mesh[mure::MeshType::overleaves][level],
    //                                          mesh[mure::MeshType::overleaves][level]);

    //     overleaves([&](auto, auto &interval, auto) {
    //         auto k = interval[0]; 

    //         std::cout<<std::endl<<"[OL Value check] Level = "<<level<<" Patch = "<<k<<std::endl<<(f(0, level, k) + f(1, level, k))<<std::flush;

    //     });
    // }




    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    Field help_f{"help_f", mesh};
    help_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        // If we are at the finest level, we no not need to correct
        if (level == max_level) {
            std::size_t j = 0; 
            double coeff = 1.;

            auto leaves = mure::intersection(mesh[mure::MeshType::cells][max_level],
                                      mesh[mure::MeshType::cells][max_level]);


            leaves([&](auto, auto &interval, auto) {

                auto k = interval[0]; 

                auto fp = xt::eval(f(0, max_level, k - 1));
                auto fm = xt::eval(f(1, max_level, k + 1));

                // COLLISION    

                auto uu = xt::eval(fp + fm);
                auto vv = xt::eval(lambda * (fp - fm));
            
                vv = (1 - s_rel) * vv + s_rel * 0.75 * uu;
                //vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                new_f(0, max_level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, max_level, k) = .5 * (uu - 1. / lambda * vv);
            });
        }

        // Otherwise, correction is needed
        else
        {

            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1); 
            double coeff = 1. / (1 << j);

            // We take the overleaves corresponding to the existing leaves
            auto overleaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                                 mesh[mure::MeshType::cells][level]).on(level + 1);
            
            overleaves([&](auto, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x

                //std::cout<<std::endl<<"Level + 1 "<<(level + 1)<<" interval = "<<k<<" Values "<<std::endl<<f(0, level + 1, k - 2)<<std::flush; 


                auto fp = xt::eval(f(0, level + 1, k));
                auto fm = xt::eval(f(1, level + 1, k));
 
                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += coeff * weight * f(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fm += coeff * weight * f(1, level + 1, k + stencil);
                }

                // Save it
                help_f(0, level + 1, k) = fp;
                help_f(1, level + 1, k) = fm;

            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            leaves([&](auto, auto &interval, auto) {
                auto k = interval[0]; 

                // Projection
                auto fp_advected = 0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1));
                auto fm_advected = 0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1));

                auto uu = xt::eval(fp_advected + fm_advected);
                auto vv = xt::eval(lambda * (fp_advected - fm_advected));
            
                vv = (1 - s_rel) * vv + s_rel * 0.75 * uu;
                //vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);

            });   
        }
    }

    std::swap(f.array(), new_f.array());
}


template<class Field, class Pred>
void one_time_step_matrix_corrected(Field &f, const Pred& pred_coeff, double s_rel)
{

    double lambda = 1.;

    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto min_level = mesh.max_level();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);

    Field new_f_final{"new_f_final", mesh};
    new_f_final.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);

        // auto expghosts = mure::intersection(mesh[mure::MeshType::cells_and_ghosts][level],
        //                                     mesh[mure::MeshType::cells_and_ghosts][level]);

        xt::xtensor_fixed<int, xt::xshape<1>> stencil_plus;
        xt::xtensor_fixed<int, xt::xshape<1>> stencil_minus;

        stencil_plus[0] = 1; // 1 is enough because it multiplies by 2
        stencil_minus[0] = -1;
        
        if (level < max_level)  {


            mure::Box<double, 1> boxfoo({-3}, {3});
            mure::Mesh< mure::MRConfig<1, 2>> meshfoo{boxfoo, min_level, max_level};
            Field f_copy{f.name(), meshfoo, f.bc()}; // This is just to create another mesh
                                                    // In order not to override the existing one
                                                    // due to the fact that most of the operations
                                                    // are applied to pointers

            refinement_up_one_level(f, f_copy, level);
            mure::mr_projection(f_copy);
            f_copy.update_bc();
            mure::mr_prediction(f_copy);
            auto mesh_copy = f_copy.mesh();



            // Does union work well ?

            auto exp_jp1 = mure::intersection(mesh[mure::MeshType::cells][level],
                                              mesh[mure::MeshType::cells][level]).on(level + 1);

            auto exp_jp1_with_ghosts = mure::union_(mure::union_(mesh[mure::MeshType::cells][level], 
                                mure::translate(mesh[mure::MeshType::cells][level], stencil_plus)),
                                mure::translate(mesh[mure::MeshType::cells][level], stencil_minus)).on(level + 1);

            // This is the set on which we have to keep the value
            // Since it is necessary to preserve the quality of the solution
            // Based on multiresolution
            auto not_to_predict = mure::intersection(mesh[mure::MeshType::cells][level + 1], exp_jp1_with_ghosts);
            auto to_predict = mure::difference(exp_jp1_with_ghosts, mesh[mure::MeshType::cells][level + 1]);


            // THe problem is that it writes and it cant
            //to_predict.apply_op(level + 1, prediction_source_destination(f, f_copy));
            // Il faudrait modifier le maillage mais ça ma l'air compliqué.
            
            // En fait les predictions sont déja faites
            
            Field new_f{"new_f", mesh_copy};
            new_f.array().fill(0.);


            exp_jp1([&](auto, auto &interval, auto) {


                std::size_t j = max_level - (level + 1); 
                double coeff = 1. / (1 << j);


                auto k = interval[0]; // Logical index in x

                std::cout<<std::endl<<"level =  "<<(level + 1)<<"  Interval = "<<k<<std::flush;


                auto fp = xt::eval(f_copy(0, level + 1, k));
                auto fm = xt::eval(f_copy(1, level + 1, k));


                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += coeff * weight * f_copy(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += coeff * weight * f_copy(1, level + 1, k + stencil);
                }

                // COLLISION    

                auto uu = xt::eval(fp + fm);
                auto vv = xt::eval(lambda * (fp - fm));

                //vv = (1 - s_rel) * vv + s_rel * 0.75 * uu;
                vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                new_f(0, level + 1, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level + 1, k) = .5 * (uu - 1. / lambda * vv);
                
                // From new_f we have to come back by projection
                auto k_original = k/2;
                k.step = 2;
                new_f_final(0, level, k_original) = 0.5 * (new_f(0, level + 1, k) + new_f(0, level + 1, k + 1)); 
            });

        }
        else {
            std::size_t j = 0; 

            auto exp = mure::intersection(mesh[mure::MeshType::cells][max_level],
                                          mesh[mure::MeshType::cells][max_level]);
            exp([&](auto, auto &interval, auto) {

            auto k = interval[0]; // Logical index in x


                auto fp = xt::eval(f(0, level, k));
                auto fm = xt::eval(f(1, level, k));
 
                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += 1.0 * weight * f(0, level, k + stencil);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += 1.0 * weight * f(1, level, k + stencil);
                }

                // COLLISION    

                auto uu = xt::eval(fp + fm);
                auto vv = xt::eval(lambda * (fp - fm));

                //vv = (1 - s_rel) * vv + s_rel * 0.75 * uu;
                vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                new_f_final(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f_final(1, level, k) = .5 * (uu - 1. / lambda * vv);
            });
        }
    }

    std::swap(f.array(), new_f_final.array());
}



template<class Field>
void one_time_step(Field &f, double s)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1.;//, s = 1.0;
    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    f.update_bc();
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

            bool cheap = false;
            
            auto fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, memoization_map, cheap)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, memoization_map, cheap));

            auto fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, memoization_map, cheap)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, memoization_map, cheap));
            
            
            // This is the CHEAP FLUX EVALUATION
            
            // auto fp = f(0, level, i); // Just to give the shape ....
            // auto fm = f(1, level, i);

            // auto exist_0_m1 = mesh.exists(level, i - 1, mure::MeshType::cells);
            // auto exist_0_p1 = mesh.exists(level, i + 1, mure::MeshType::cells);

            // auto exist_up_m1 = mesh.exists(level + 1, 2*i - 1, mure::MeshType::cells);
            // auto exist_up_p1 = mesh.exists(level + 1, 2*i + 1, mure::MeshType::cells);

            // auto exist_down_m1 = mesh.exists(level - 1, i/2 - 1, mure::MeshType::cells); // Verify this division by 2... sometimes is problematic
            // auto exist_down_p1 = mesh.exists(level - 1, i/2 + 1, mure::MeshType::cells);

            // // The left neigh at the same level exists
            // xt::masked_view(fp, exist_0_m1) = (1.0 - coeff) * f(0, level, i) + coeff * f(0, level, i - 1);
            // // The right neigh at the same level exists
            // xt::masked_view(fm, exist_0_p1) = (1.0 - coeff) * f(1, level, i) + coeff * f(1, level, i + 1);


            // // This is problematic... ASK
            // xt::masked_view(xt::masked_view(fp, !exist_0_m1), 
            //                                     exist_up_m1) = (1.0 - coeff) * f(0, level, i) + coeff * f(0, level + 1, 2*i - 1); 
                                                
            // xt::masked_view(xt::masked_view(fm, !exist_0_p1), 
            //                                         exist_up_p1) = (1.0 - coeff) * f(1, level, i);// + coeff * f(1, level + 1, 2*i + 1); 


            


            // COLLISION    

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            
            //vv = (1 - s) * vv + s * 0.75 * uu;

            vv = (1 - s) * vv + s * .5 * uu * uu;

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

template<class Field>
void save_refined_solution(Field &f, std::size_t min_level, std::size_t max_level, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();

    std::stringstream str;
    str << "LBM_D1Q2_Burgers_refined_solution_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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

// template<class Field, class FieldTag>
// double compute_error(const Field & f, const FieldTag & tag, double t)
// {
//     double error_to_return = 0.0;

//     // Getting ready for memoization
//     using interval_t = typename Field::Config::interval_t;
//     std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
//     memoization_map.clear();


//     auto mesh = f.mesh();
//     auto max_level = mesh.max_level();

//     double dx = 1 << max_level;


//     Field error_cell_by_cell{"error_cell_by_cell", mesh};
//     error_cell_by_cell.array().fill(0.);


//     auto subset = intersection(mesh.initial_mesh(), mesh.initial_mesh()).on(max_level);
//     subset([&](auto, auto &interval, auto) {
//         auto i = interval[0];


//         std::cout<<"\n\nHere  "<<i;

//         auto fp = prediction(f, max_level, 0, i, 0, tag, memoization_map); // BUG ICI
//         //auto fm = prediction(f, max_level, 0, i, 1, tag, memoization_map);

//         // CELA IL FAUT LE TRAITER MAIS ON VERRA APRES...

//         // auto rho = xt::eval(fp + fm);

//         // error_cell_by_cell(0, max_level, i) = dx * xt::abs(rho);
//     });

//     return error_to_return;
    
    
//     //return xt::sum(xt::view(error_cell_by_cell, 0, max_level, xt::all));
// }

// template<class Field, class FieldR>
template<class Config, class FieldR>
std::array<double, 2> compute_error(mure::Field<Config, double, 2> &f, FieldR & fR, double t)
{


    auto mesh = f.mesh();

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    fR.update_bc();    

    mure::mr_projection(f);    
    f.update_bc(); // Important especially when we enforce Neumann...for the Riemann problem
    mure::mr_prediction(f);  // C'est supercrucial de le faire.


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
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("7"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
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

            // std::array<double, 4> foo1 {1, 0, 0, 0};
            // auto foo = mult_by_prediction_matrix(2, foo1);

            // for (auto el : foo){
            //     std::cout<<std::endl<<el;
            // }

            // return 0;





            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 1;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();
            double s = result["s"].as<double>();


            mure::Box<double, dim> box({-3}, {3});
            mure::Mesh<Config> mesh{box, min_level, max_level};
            mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

    
            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);


            // Initialization
            auto f  = init_f(mesh , 0.0);
            auto fR = init_f(meshR, 0.0);             

            double T = 1.2;
            double dx = 1.0 / (1 << max_level);
            double dt = dx;

            std::size_t N = static_cast<std::size_t>(T / dt);

            double t = 0.0;

            std::ofstream out_time_frames;
            std::ofstream out_error_exact_ref;
            std::ofstream out_diff_ref_adap;
            std::ofstream out_compression;

            out_time_frames.open     ("./d1q2/time_frame_s_"     +std::to_string(s)+"_eps_"+std::to_string(eps)+".dat");
            out_error_exact_ref.open ("./d1q2/error_exact_ref_"  +std::to_string(s)+"_eps_"+std::to_string(eps)+".dat");
            out_diff_ref_adap.open   ("./d1q2/diff_ref_adap_s_"  +std::to_string(s)+"_eps_"+std::to_string(eps)+".dat");
            out_compression.open     ("./d1q2/compression_s_"    +std::to_string(s)+"_eps_"+std::to_string(eps)+".dat");


            for (std::size_t nb_ite = 0; nb_ite < 2; ++nb_ite)
            {
                tic();
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    //std::cout<<std::endl<<"Passe "<<i;
                    if (coarsening(f, eps, i))
                        break;
                }



                auto duration_coarsening = toc();

                // save_solution(f, eps, nb_ite, "coarsening");

                tic();
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Refinement "<<i<<std::flush;
                    if (refinement(f, eps, 0.0, i))
                        break;
                }

                if(nb_ite == 1) {

                    for (std::size_t level = 0; level <= max_level; ++level)    {
                        auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                                         mesh[mure::MeshType::cells][level]);
            
                        leaves([&](auto, auto &interval, auto) {
                            auto k = interval[0];
                            std::cout<<std::endl<<"Check vales after refinement - Level = "<<level<<" Cell = "<<k<<std::endl<<(f(0, level, k) + f(1, level, k));
                        });
                    }
                }







                auto duration_refinement = toc();
                save_solution(f, eps, nb_ite, "refinement");


                auto error = compute_error(f, fR, t);

                out_time_frames    <<t       <<std::endl;
                out_error_exact_ref<<error[0]<<std::endl;
                out_diff_ref_adap  <<error[1]<<std::endl;
                out_compression    <<static_cast<double>(mesh.nb_cells(mure::MeshType::cells)) 
                                   / static_cast<double>(meshR.nb_cells(mure::MeshType::cells))<<std::endl;

                save_refined_solution(fR, min_level, max_level, eps, nb_ite);

                std::cout<<std::endl<<"Mesh after refinement and addition of the overleaves"<<std::endl<<f.mesh();


        
                tic();
                //one_time_step(f, s);
                //one_time_step_matrix(f, pred_coeff, s);
                //one_time_step_matrix_corrected(f, pred_coeff, s);
                one_time_step_matrix_overleaves(f, pred_coeff, s);

                auto duration_scheme = toc();

                tic();
                one_time_step(fR, s);
                auto duration_schemeR = toc();

                t += dt;

                tic();
                save_solution(f, eps, nb_ite, "onetimestep");
                auto duration_save = toc();


                std::cout<<std::endl<<"\n=======Iteration "<<nb_ite<<"  time "<<t<<" summary========"
                                    <<"\nCoarsening: "<<duration_coarsening
                                    <<"\nRefinement: "<<duration_refinement
                                    <<"\nScheme: "<<duration_scheme
                                    <<"\nScheme reference: "<<duration_schemeR
                                    <<"\nSave: "<<duration_save
                                    <<"\nError exact - referece = "<< error[0]
                                    <<"\nError adaptive - referece = "<< error[1] << "\n";

                                    

            }
            
            out_time_frames.close();
            out_error_exact_ref.close();
            out_diff_ref_adap.close();
            out_compression.close();

        }

    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
