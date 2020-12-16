#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "harten.hpp"
#include "prediction_map_2d.hpp"

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

// double lambda = 1./0.3; //4.0; // We always used it
// double lambda = 1./0.2499; //4.0;

double lambda = 1./0.2;

double sigma_q = 0.5; 
double sigma_xy = 0.5;

// double sq = 1.6;// For the sod tube
double sq = 1.75;//1./(.5 + sigma_q);
// double sxy = 0.5;//1./(.5 + sigma_xy);  
double sxy = 1.5;//1./(.5 + sigma_xy);  

        
double gm = 1.4;



template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 16;
    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0}
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double rho = 1.0; // Density
        double qx = 0.0; // x-momentum
        double qy = 0.0; // y-momentum
        double e = 0.0;

        double p = 1.0;

        // rho = (y < 0.5 + 0.002 * cos(2.*M_PI*2*x)) ? 0.5 : 1.0; 
        // qx  = rho * (y < 0.5 + 0.025 * sin(2*M_PI*2*x)) ? -0.5 : 0.5; 
        // // qx = - 0.5 * rho * (tanh(60 * (y - 0.5 + 0.025 * sin(2*M_PI*4*x))));
        // qy = -0.05;
        // p = (y < 0.5 + 0.002 * cos(2.*M_PI*2*x)) ? 0.5 : 0.1;

        if (x < 0.5)    {
            if (y < 0.5)    {
                // 3
                // rho = 1.0;
                // qx = rho * 0.75;
                // qy = rho * 0.5;
                // double p = 1.0;

                // rho = 0.5;
                // qx = rho * 0.0;
                // qy = rho * 0.0;
                // p = 1.0;

                // // Configuration 11
                // rho = 0.8;
                // qx = rho * 0.1;
                // qy = rho * 0.0;
                // p = 0.4;


                // // Configuration 12
                rho = 0.8;
                qx = rho * 0.0;
                qy = rho * 0.0;
                p = 1.;

                //// Configuration 17
                // rho = 1.0625;
                // qx = rho * 0.;
                // qy = rho * 0.2145;
                // p = 0.4;

                //// Configuration 3
                // rho = 0.138;
                // qx = rho * 1.206;
                // qy = rho * 1.206;
                // p = 0.029;

            }
            else
            {
                // 2   
                // rho = 2.0;
                // qx = rho * (-0.75);
                // qy = rho * (0.5);
                // double p = 1.0;

                // rho = 0.5;
                // qx = rho * 0.7276;
                // qy = rho * 0.0;
                // p = 1.0;

                // // Configuration 11

                // rho = 0.5313;
                // qx = rho * 0.8276;
                // qy = rho * 0.0;
                // p = 0.4;


                // // Configuration 12

                rho = 1.;
                qx = rho * 0.7276;
                qy = rho * 0.0;
                p = 1.;


                //// Configuration 17
                // rho = 2.;
                // qx = rho * 0.;
                // qy = rho * (-0.3);
                // p = 1.;

                // // Configuration 3
                // rho = 0.5323;
                // qx = rho * 1.206;
                // qy = rho * 0.;
                // p = 0.3;
            }
        }
        else
        {
            if (y < 0.5)    {
                // 4
                // rho = 3.0;
                // qx = rho * (0.75);
                // qy = rho * (-0.5);
                // double p = 1.0;

                // rho = 1.0;
                // qx = rho * 0.0;
                // qy = rho * 0.7276;
                // p = 1.0;

                // // Configuration 11
                // rho = 0.5313;
                // qx = rho * 0.1;
                // qy = rho * 0.7276;
                // p = 0.4;

                // // // Configuration 12
                rho = 1.;
                qx = rho * 0.0;
                qy = rho * 0.7276;
                p = 1.;

                //// Configuration 17
                // rho = 0.5197;
                // qx = rho * 0.;
                // qy = rho * (-1.1259);
                // p = 0.4;

                //// Configuration 3
                // rho = 0.5323;
                // qx = rho * 0.;
                // qy = rho * 1.206;
                // p = 0.3;
            }
            else
            {
                // 1
                // rho = 1.0;
                // qx = rho * (-0.75);
                // qy = rho * (-0.5);
                // double p = 1.0;
                
                // rho = 0.5313;
                // qx = rho * 0.0;
                // qy = rho * 0.0;
                // p = 0.4;

                // // Configuration 11
                // rho = 1.;
                // qx = rho * 0.1;
                // qy = rho * 0.0;
                // p = 1.;

                // // // Configuration 12
                rho = 0.5313;
                qx = rho * 0.0;
                qy = rho * 0.0;
                p = 0.4;


                //// Configuration 17
                // rho = 1.;
                // qx = rho * 0.;
                // qy = rho * (-0.4);
                // p = 1.;

                //// Configuration 3
                // rho = 1.5;
                // qx = rho * 0.;
                // qy = rho * 0.;
                // p = 1.5;
            }
        }

        // A sort of sod shock tube

        // if (x < 0.5)    {
        //     rho = 1.;
        //     qx = 0.;
        //     qy = 0.;
        //     p = 1.;
        // }
        // else
        // {
        //     rho = 0.125;
        //     qx = 0.;
        //     qy = 0.;
        //     p = 0.1;
        // }
        
        
        
        e = p / (gm - 1.) + 0.5 * (qx*qx + qy*qy) / rho;     

        // Conserved momenti
        double m0_0 = rho;
        double m1_0 = qx;
        double m2_0 = qy;
        double m3_0 = e;

        // Non conserved at equilibrium
        double m0_1 = m1_0;
        double m0_2 = m2_0;
        double m0_3 = 0.0;

        double m1_1 =     (3./2. - gm/2.) * (m1_0*m1_0)/(m0_0)
                        + (1./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (gm - 1.) * m3_0;
        double m1_2 = m1_0*m2_0/m0_0;
        double m1_3 = 0.0;

        double m2_1 = m1_0*m2_0/m0_0;

        double m2_2 =     (3./2. - gm/2.) * (m2_0*m2_0)/(m0_0)
                        + (1./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (gm - 1.) * m3_0;
        double m2_3 = 0.0;

        double m3_1 = gm*(m1_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m1_0*m1_0*m1_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m1_0*m2_0*m2_0)/(m0_0*m0_0);
        double m3_2 = gm*(m2_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m2_0*m2_0*m2_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m2_0*m1_0*m1_0)/(m0_0*m0_0);
        double m3_3 = 0.0;

        // We come back to the distributions
        f[cell][0] = .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
        f[cell][1] = .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
        f[cell][2] = .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
        f[cell][3] = .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;

        f[cell][4] = .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
        f[cell][5] = .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
        f[cell][6] = .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
        f[cell][7] = .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;

        f[cell][8]  = .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
        f[cell][9]  = .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
        f[cell][10] = .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
        f[cell][11] = .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;

        f[cell][12] = .25 * m3_0 + .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
        f[cell][13] = .25 * m3_0                    + .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;
        f[cell][14] = .25 * m3_0 - .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
        f[cell][15] = .25 * m3_0                    - .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;

    });

    return f;
}

// template<class coord_index_t>
// auto compute_prediction(std::size_t min_level, std::size_t max_level)
// {
//     coord_index_t i = 0, j = 0;
//     std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

//     for(std::size_t k=0; k<max_level-min_level+1; ++k)
//     {
//         int size = (1<<k);
//         data[k].resize(8);
//         for (int l = 0; l < size; ++l)
//         {
//             // Be careful, there is no sign on this fluxes

//             // Along x (vertical edge)
//             data[k][0] += prediction(k, i*size - 1, j*size + l); // In W
//             data[k][1] += prediction(k, (i+1)*size - 1, j*size + l); // Out E
//             data[k][2] += prediction(k, i*size + l, j*size - 1); // In S
//             data[k][3] += prediction(k, i*size + l, (j+1)*size - 1); // Out N

//             // Along y (horizontal edge)
//             data[k][4] += prediction(k, (i+1)*size, j*size + l); // In E
//             data[k][5] += prediction(k, i*size, j*size + l); // Out W
//             data[k][6] += prediction(k, i*size + l, (j+1)*size); // In N
//             data[k][7] += prediction(k, i*size + l, j*size); // Out S
//         }
//     }
//     return data;
// }


template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);


    auto rotation_of_pi_over_two = [] (int alpha, int k, int h)
    {
        // Returns the rotation of (k, h) of an angle alpha * pi / 2.
        // All the operations are performed on integer, to be exact

        int cosinus = static_cast<int>(std::round(std::cos(alpha * M_PI / 2.)));
        int sinus   = static_cast<int>(std::round(std::sin(alpha * M_PI / 2.)));

        return std::pair<int, int> (cosinus * k - sinus   * h, 
                                      sinus * k + cosinus * h);
    };

    // Transforms the coordinates to apply the rotation
    auto tau = [] (int delta, int k)
    {
        // The case in which delta = 0 is rather exceptional
        if (delta == 0) {
            return k;
        }
        else {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < tmp) ? (k - tmp) : (k - tmp + 1));
        }
    };

    auto tau_inverse = [] (int delta, int k)
    {
        if (delta == 0) {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < 0) ? (k + tmp) : (k + tmp - 1));   
        }
    };

    for(std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1<<k);


        data[k].resize(12);


        // Parallel velocities
        
        for (int alpha = 0; alpha <= 3; ++alpha)
        {

            // std::
            
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are computed
                // is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][0 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_in.first ), tau_inverse(k, rotated_in.second ));
                data[k][1 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));

                // For the cells inside the domain, we can already
                // Combine entering and exiting fluxes and 
                // we have a compensation of many cells.
                data[k][8 + alpha] += (prediction(k, tau_inverse(k, rotated_in.first ), tau_inverse(k, rotated_in.second ))
                                     - prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second)));

            }
        }

    }
    return data;
}


// I do many separate functions because the return type
// is not necessarely the same between directions and I want to avoid 
// using a template, which indeed comes back to the same than this.
template<class Mesh> 
auto get_adjacent_boundary_east(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level); // When we are not at the finest level, we must translate more

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), - xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SE
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_north(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing NW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_west(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NW
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_south(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing SE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d11)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), - coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d1m1)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d11)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d1m1)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), -coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}





// We have to average only the fluxes
template<class Field, class pred>
void one_time_step_overleaves_corrected(Field &f, const pred& pred_coeff, std::size_t iter)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();
    
    mure::mr_projection(f);
    // if (max_level != min_level && iter == 105){
    //     std::stringstream s;
    //     s << "before_LB_scheme_projection_"<<iter;
    //     auto h5file = mure::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, f);
    // }    
    f.update_bc(); // It is important to do so
    // if (max_level != min_level && iter == 105){
    //     std::stringstream s;
    //     s << "before_LB_scheme_update_bc_"<<iter;
    //     auto h5file = mure::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, f);
    // }    
    mure::mr_prediction(f);
    // if (max_level != min_level && iter == 105){
    //     std::stringstream s;
    //     s << "before_LB_scheme_prediction_"<<iter;
    //     auto h5file = mure::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, f);
    // }    
    // f.update_bc(); // It is important to do so

    mure::mr_prediction_overleaves(f);
    // f.update_bc(); // It is important to do so

    // if (max_level != min_level && iter == 105){
    //     std::stringstream s;
    //     s << "before_LB_scheme_overleaves_"<<iter;
    //     auto h5file = mure::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, f);
    // }    

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    // This stored the fluxes computed at the level
    // of the overleaves
    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);


    Field advected{"advected", mesh};
    advected.array().fill(0.);


    double time_advection_overleaves_boundary = 0.;
    double time_advection_overleaves_inside = 0.;
    double time_collision_overleaves = 0.;

    spdlog::stopwatch sw;


    for (std::size_t level = min_level; level <= max_level; ++level)
    {

        double coeff = 1./(1 << (max_level - level));

        auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                         mesh[mure::MeshType::cells][level]); 

        leaves([&](auto& index, auto &interval, auto) {
            auto k = interval[0]; // Logical index in x
            auto h = index[0];    // Logical index in y 
            
            for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                advected(0 + 4 * scheme_n, level, k, h) =  (1. - coeff) * f(0 + 4 * scheme_n, level, k, h) + coeff * f(0 + 4 * scheme_n, level, k - 1, h    );
                advected(1 + 4 * scheme_n, level, k, h) =  (1. - coeff) * f(1 + 4 * scheme_n, level, k, h) + coeff * f(1 + 4 * scheme_n, level, k,     h - 1);
                advected(2 + 4 * scheme_n, level, k, h) =  (1. - coeff) * f(2 + 4 * scheme_n, level, k, h) + coeff * f(2 + 4 * scheme_n, level, k + 1, h    ); 
                advected(3 + 4 * scheme_n, level, k, h) =  (1. - coeff) * f(3 + 4 * scheme_n, level, k, h) + coeff * f(3 + 4 * scheme_n, level, k,     h + 1);
            }


            auto f0 = xt::eval(advected(0, level, k, h));
            auto f1 = xt::eval(advected(1, level, k, h));
            auto f2 = xt::eval(advected(2, level, k, h));
            auto f3 = xt::eval(advected(3, level, k, h));
                
            auto f4 = xt::eval(advected(4, level, k, h));
            auto f5 = xt::eval(advected(5, level, k, h));
            auto f6 = xt::eval(advected(6, level, k, h));
            auto f7 = xt::eval(advected(7, level, k, h));

            auto f8  = xt::eval(advected(8,  level, k, h));
            auto f9  = xt::eval(advected(9,  level, k, h));
            auto f10 = xt::eval(advected(10, level, k, h));
            auto f11 = xt::eval(advected(11, level, k, h));
 
            auto f12 = xt::eval(advected(12, level, k, h));
            auto f13 = xt::eval(advected(13, level, k, h));
            auto f14 = xt::eval(advected(14, level, k, h));
            auto f15 = xt::eval(advected(15, level, k, h));

            // We compute the advected momenti
            auto m0_0 = xt::eval(                 f0 + f1 + f2 + f3) ;
            auto m0_1 = xt::eval(lambda        * (f0      - f2      ));
            auto m0_2 = xt::eval(lambda        * (     f1      - f3));
            auto m0_3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));

            auto m1_0 = xt::eval(                 f4 + f5 + f6 + f7) ;
            auto m1_1 = xt::eval(lambda        * (f4      - f6      ));
            auto m1_2 = xt::eval(lambda        * (     f5      - f7));
            auto m1_3 = xt::eval(lambda*lambda * (f4 - f5 + f6 - f7));

            auto m2_0 = xt::eval(                 f8 + f9 + f10 + f11) ;
            auto m2_1 = xt::eval(lambda        * (f8      - f10      ));
            auto m2_2 = xt::eval(lambda        * (     f9       - f11));
            auto m2_3 = xt::eval(lambda*lambda * (f8 - f9 + f10 - f11));

            auto m3_0 = xt::eval(                 f12 + f13 + f14 + f15) ;
            auto m3_1 = xt::eval(lambda        * (f12       - f14      ));
            auto m3_2 = xt::eval(lambda        * (      f13       - f15));
            auto m3_3 = xt::eval(lambda*lambda * (f12 - f13 + f14 - f15));


            m0_1 = (1 - sq) *  m0_1 + sq * (m1_0);
            m0_2 = (1 - sq) *  m0_2 + sq * (m2_0);
            m0_3 = (1 - sxy) * m0_3; 


            m1_1 = (1 - sq) *  m1_1 + sq * ((3./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (1./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (gm - 1.) * m3_0);
            m1_2 = (1 - sq) *  m1_2 + sq * (m1_0*m2_0/m0_0);
            m1_3 = (1 - sxy) * m1_3; 

            m2_1 = (1 - sq) *  m2_1 + sq * (m1_0*m2_0/m0_0);
            m2_2 = (1 - sq) *  m2_2 + sq * ((3./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (1./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (gm - 1.) * m3_0);
            m2_3 = (1 - sxy) * m2_3; 

            m3_1 = (1 - sq) *  m3_1 + sq * (gm*(m1_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m1_0*m1_0*m1_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m1_0*m2_0*m2_0)/(m0_0*m0_0));
            m3_2 = (1 - sq) *  m3_2 + sq * (gm*(m2_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m2_0*m2_0*m2_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m2_0*m1_0*m1_0)/(m0_0*m0_0));
            m3_3 = (1 - sxy) * m3_3; 


            new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
            new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
            new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
            new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;

            new_f(4, level, k, h) =  .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
            new_f(5, level, k, h) =  .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
            new_f(6, level, k, h) =  .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
            new_f(7, level, k, h) =  .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;

            new_f(8, level, k, h)  =  .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
            new_f(9, level, k, h)  =  .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
            new_f(10, level, k, h) =  .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
            new_f(11, level, k, h) =  .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;

            new_f(12, level, k, h) =  .25 * m3_0 + .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
            new_f(13, level, k, h) =  .25 * m3_0                    + .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;
            new_f(14, level, k, h) =  .25 * m3_0 - .5/lambda * (m3_1)                    + .25/(lambda*lambda) * m3_3;
            new_f(15, level, k, h) =  .25 * m3_0                    - .5/lambda * (m3_2) - .25/(lambda*lambda) * m3_3;
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
    str << "LBM_D2Q4_3_Euler_direct_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> rho{"rho", mesh};
    mure::Field<Config> qx{"qx", mesh};
    mure::Field<Config> qy{"qy", mesh};
    mure::Field<Config> e{"e", mesh};
    mure::Field<Config> s{"entropy", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
        qx[cell]  = f[cell][4] + f[cell][5] + f[cell][6] + f[cell][7];
        qy[cell]  = f[cell][8] + f[cell][9] + f[cell][10] + f[cell][11];
        e[cell]   = f[cell][12] + f[cell][13] + f[cell][14] + f[cell][15];

        // Computing the entropy with multiplicative constant 1 and additive constant 0
        auto p = (gm - 1.) * (e[cell] - .5 * (std::pow(qx[cell], 2.) + std::pow(qy[cell], 2.)) / rho[cell]);
        s[cell] = xt::log(p / xt::pow(rho[cell], gm));

    });
    h5file.add_field(rho);
    h5file.add_field(qx);
    h5file.add_field(qy);
    h5file.add_field(e);
    h5file.add_field(s);

    h5file.add_field(f);
    h5file.add_field(level_);
}



// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template<class Field, class interval_t, class ordinates_t, class ordinates_t_bis>
xt::xtensor<double, 2> prediction_all(const Field & f, std::size_t level_g, std::size_t level, 
                                      const interval_t & k, const ordinates_t & h, 
                                      std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>, xt::xtensor<double, 2>> & mem_map)
{

    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    // mem_map.clear();

    auto it = mem_map.find({level_g, level, k, h});


    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())    {

        return it->second;
    }
    else
    {
        

    auto mesh = f.mesh();

    // We put only the size in x (k.size()) because in y
    // we only have slices of size 1. 
    // The second term (1) should be adapted according to the 
    // number of fields that we have.
    // std::vector<std::size_t> shape_x = {k.size(), 4};
    std::vector<std::size_t> shape_x = {k.size(), 16};
    xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(mure::MeshType::cells_and_ghosts, level_g + level, k, h); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);
        
    // for (int h_field = 0; h_field < 4; ++h_field)  {
    for (int h_field = 0; h_field < 16; ++h_field)  {
        xt::view(mask_all, xt::all(), h_field) = mask;
    }    

    // Recursion finished
    if (xt::all(mask))
    {                 
        // return xt::eval(f(0, 4, level_g + level, k, h));
        return xt::eval(f(0, 16, level_g + level, k, h));

    }

    // If we cannot stop here

    auto kg = k >> 1;
    kg.step = 1;

    xt::xtensor<double, 2> val = xt::empty<double>(shape_x);


    /*
    --------------------
    NW   |   N   |   NE
    --------------------
     W   | EARTH |   E
    --------------------
    SW   |   S   |   SE
    --------------------
    */


    auto earth  = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1)    , mem_map));
    auto W      = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1)    , mem_map));
    auto E      = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1)    , mem_map));
    auto S      = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1) - 1, mem_map));
    auto N      = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1) + 1, mem_map));
    auto SW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) - 1, mem_map));
    auto SE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) - 1, mem_map));
    auto NW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) + 1, mem_map));
    auto NE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) + 1, mem_map));


    // This is to deal with odd/even indices in the x direction
    std::size_t start_even = (k.start & 1) ?     1         :     0        ; 
    std::size_t start_odd  = (k.start & 1) ?     0         :     1        ; 
    std::size_t end_even   = (k.end & 1)   ? kg.size()     : kg.size() - 1;
    std::size_t end_odd    = (k.end & 1)   ? kg.size() - 1 : kg.size()    ;

    int delta_y = (h & 1) ? 1 : 0;
    int m1_delta_y = (delta_y == 0) ? 1 : -1; // (-1)^(delta_y) 

    // We recall the formula before doing everything
    /*
    f[j + 1][2k + dx][2h + dy] = f[j][k][h] + 1/8 * (-1)^dx * (f[j][k - 1][h] - f[j][k + 1][h])
                                            + 1/8 * (-1)^dy * (f[j][k][h - 1] - f[j][k][h + 1])
                                - 1/64 * (-1)^(dx+dy) * (f[j][k + 1][h + 1] - f[j][k - 1][h + 1]
                                                         f[j][k - 1][h - 1] - f[j][k + 1][h - 1])

    dx = 0, 1
    dy = 0, 1
    */

    
    xt::view(val, xt::range(start_even, _, 2)) = xt::view(                        earth 
                                                          + 1./8               * (W - E) 
                                                          + 1./8  * m1_delta_y * (S - N) 
                                                          - 1./64 * m1_delta_y * (NE - NW - SE + SW), xt::range(start_even, _));



    xt::view(val, xt::range(start_odd, _, 2))  = xt::view(                        earth 
                                                          - 1./8               * (W - E) 
                                                          + 1./8  * m1_delta_y * (S - N)
                                                          + 1./64 * m1_delta_y * (NE - NW - SE + SW), xt::range(_, end_odd));

    xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

    for(int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
    {
        if (mask[k_mask])
        {
            // xt::view(out, k_mask) = xt::view(f(0, 4, level_g + level, {k_int, k_int + 1}, h), 0);
            xt::view(out, k_mask) = xt::view(f(0, 16, level_g + level, {k_int, k_int + 1}, h), 0);

        }
    }

    // It is crucial to use insert and not []
    // in order not to update the value in case of duplicated (same key)
    mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>{level_g, level, k, h}
                                  ,out));


    return out;

    }
}



template<class Field, class FieldFull>
double compute_error(Field & f, FieldFull & f_full)
{

    
    auto mesh = f.mesh();
    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto init_mesh = f_full.mesh();


    using Config = typename FieldFull::Config;


    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);



    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0}
                    }} };

  
    // mure::Field<Config, double, 4> f_reconstructed("f_reconstructed", init_mesh, bc);
    mure::Field<Config, double, 16> f_reconstructed("f_reconstructed", init_mesh, bc);

    f_reconstructed.array().fill(0.);


    // For memoization
    using interval_t  = typename Config::interval_t; // Type in X
    using ordinates_t = typename Config::index_t;    // Type in Y
    std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t>, xt::xtensor<double, 2>> memoization_map;

    memoization_map.clear();

    double error = 0.;
    double norm = 0.;

    double dx = 1. / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves_on_finest = mure::intersection(mesh[mure::MeshType::cells][level],
                                                    mesh[mure::MeshType::cells][level]);
            
        leaves_on_finest.on(max_level)([&](auto& index, auto &interval, auto) {
            auto k = interval[0];
            auto h = index[0];

            f_reconstructed(max_level, k, h) = prediction_all(f, level, max_level - level, k, h, memoization_map);

            // Error on the density field

            auto rho_reconstructed = f_reconstructed(0, max_level, k, h) 
                                   + f_reconstructed(1, max_level, k, h)
                                   + f_reconstructed(2, max_level, k, h)
                                   + f_reconstructed(3, max_level, k, h);

            auto rho_full =  f_full(0, max_level, k, h)
                           + f_full(1, max_level, k, h)
                           + f_full(2, max_level, k, h)
                           + f_full(3, max_level, k, h);

            error += xt::sum(xt::abs(rho_reconstructed - rho_full))[0];

            norm += xt::sum(xt::abs(rho_full))[0];

        });
    }


    return (error / norm);

}


template<class Field, class FieldFull>
void save_reconstructed(Field & f, FieldFull & f_full, 
                        double eps, std::size_t ite, std::string ext="")
{

    
    auto mesh = f.mesh();
    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto init_mesh = f_full.mesh();


    using Config = typename FieldFull::Config;


    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);



    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0}
                    }} };

  
    // mure::Field<Config, double, 4> f_reconstructed("f_reconstructed", init_mesh, bc);
    mure::Field<Config, double, 16> f_reconstructed("f_reconstructed", init_mesh, bc); // To reconstruct all and see entropy
    f_reconstructed.array().fill(0.);

    mure::Field<Config> rho_reconstructed{"rho_reconstructed", init_mesh};  
    mure::Field<Config> qx_reconstructed{"qx_reconstructed", init_mesh};    
    mure::Field<Config> qy_reconstructed{"qy_reconstructed", init_mesh};    
    mure::Field<Config> E_reconstructed{"E_reconstructed", init_mesh};    
    mure::Field<Config> s_reconstructed{"s_reconstructed", init_mesh};    
  
    mure::Field<Config> rho{"rho", init_mesh};
    mure::Field<Config> qx{"qx", init_mesh};
    mure::Field<Config> qy{"qy", init_mesh};
    mure::Field<Config> E{"E", init_mesh};
    mure::Field<Config> s{"s", init_mesh};


    // For memoization
    using interval_t  = typename Config::interval_t; // Type in X
    using ordinates_t = typename Config::index_t;    // Type in Y
    std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t>, xt::xtensor<double, 2>> memoization_map;

    memoization_map.clear();

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto number_leaves = mesh.nb_cells(level, mure::MeshType::cells);


        auto leaves_on_finest = mure::intersection(mesh[mure::MeshType::cells][level],
                                                    mesh[mure::MeshType::cells][level]);
            
        leaves_on_finest.on(max_level)([&](auto& index, auto &interval, auto) {
            auto k = interval[0];
            auto h = index[0];


            f_reconstructed(max_level, k, h) = prediction_all(f, level, max_level - level, k, h, memoization_map);
            rho_reconstructed(max_level, k, h) = f_reconstructed(0, max_level, k, h)
                                               + f_reconstructed(1, max_level, k, h)
                                               + f_reconstructed(2, max_level, k, h)
                                               + f_reconstructed(3, max_level, k, h);

            qx_reconstructed(max_level, k, h) =  f_reconstructed(4, max_level, k, h)
                                               + f_reconstructed(5, max_level, k, h)
                                               + f_reconstructed(6, max_level, k, h)
                                               + f_reconstructed(7, max_level, k, h);

            qy_reconstructed(max_level, k, h) =  f_reconstructed(8, max_level, k, h)
                                               + f_reconstructed(9, max_level, k, h)
                                               + f_reconstructed(10, max_level, k, h)
                                               + f_reconstructed(11, max_level, k, h);

            E_reconstructed(max_level, k, h) =   f_reconstructed(12, max_level, k, h)
                                               + f_reconstructed(13, max_level, k, h)
                                               + f_reconstructed(14, max_level, k, h)
                                               + f_reconstructed(15, max_level, k, h);

            s_reconstructed(max_level, k, h) = xt::log(((gm - 1.) * (E_reconstructed(max_level, k, h) 
                                                                   - .5 * (xt::pow(qx_reconstructed(max_level, k, h), 2.) 
                                                                         + xt::pow(qy_reconstructed(max_level, k, h), 2.)) / rho_reconstructed(max_level, k, h))) / 
                                                        xt::pow(rho_reconstructed(max_level, k, h), gm));


            rho(max_level, k, h) = f_full(0, max_level, k, h)
                                 + f_full(1, max_level, k, h)
                                 + f_full(2, max_level, k, h)
                                 + f_full(3, max_level, k, h);

            qx(max_level, k, h) =  f_full(4, max_level, k, h)
                                 + f_full(5, max_level, k, h)
                                 + f_full(6, max_level, k, h)
                                 + f_full(7, max_level, k, h);

            qy(max_level, k, h) =  f_full(8, max_level, k, h)
                                 + f_full(9, max_level, k, h)
                                 + f_full(10, max_level, k, h)
                                 + f_full(11, max_level, k, h);    

            E(max_level, k, h) =   f_full(12, max_level, k, h)
                                 + f_full(13, max_level, k, h)
                                 + f_full(14, max_level, k, h)
                                 + f_full(15, max_level, k, h);

            s(max_level, k, h) = xt::log(((gm - 1.) * (E(max_level, k, h) 
                                                      - .5 * (xt::pow(qx(max_level, k, h), 2.) 
                                                            + xt::pow(qy(max_level, k, h), 2.)) / rho(max_level, k, h))) / 
                                           xt::pow(rho(max_level, k, h), gm));

        });
    }


    std::cout<<std::endl;

    std::stringstream str;
    str << "Euler_direct_Reconstruction_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(init_mesh);
    h5file.add_field(rho_reconstructed);
    h5file.add_field(s_reconstructed);

    h5file.add_field(rho);
    h5file.add_field(s);


}




int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Euler",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar advection equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("7"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("ite", "number of iteration", cxxopts::value<std::size_t>()->default_value("100"))
                       ("reg", "regularity", cxxopts::value<double>()->default_value("0."))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {

            //auto save_string = std::string("bruteforce");
            auto save_string = std::string("overleaves");


            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn},
                                                               {"info", spdlog::level::info}};
            constexpr size_t dim = 2;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            std::size_t total_nb_ite = result["ite"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();
            double regularity = result["reg"].as<double>();

            mure::Box<double, dim> box({0, 0}, {1, 1});
            mure::Mesh<Config> mesh{box, min_level, max_level};
            // mure::Mesh<Config> mesh_old{box, min_level, max_level};
            mure::Mesh<Config> mesh_ref{box, max_level, max_level};


            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);


            // Initialization
            auto f     = init_f(mesh,     0);
            auto f_ref = init_f(mesh_ref, 0);

            double T = 0.25;//0.3;//1.2;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);




            int N_saves = 20;
            int howoften = 1;//N / N_saves;

            for (std::size_t nb_ite = 0; nb_ite <= N; ++nb_ite)
            {
                std::cout<<std::endl<<"Iteration number = "<<nb_ite<<std::endl;

                tic();
                if (max_level > min_level)  {



                    auto mesh_old = mesh;
                    mure::Field<Config, double, 16> f_old{"u", mesh_old};
                    f_old.array() = f.array();
                    for (std::size_t i=0; i<max_level-min_level; ++i)
                    {
                        std::cout<<std::endl<<"Step "<<i<<std::flush;
                        if (harten(f, f_old, eps, regularity, i, nb_ite))
                            break;
                    }

                }

                save_solution(f, eps, nb_ite);

                auto time_mesh_adaptation = toc();

                if (nb_ite == N)    {
                    auto error_density = compute_error(f, f_ref);
                    std::cout<<std::endl<<"#### Epsilon = "<<eps<<"   error = "<<error_density<<std::flush;
                    save_reconstructed(f, f_ref, eps, 0);
                    save_solution(f, eps, 0, save_string+std::string("PAPER")); // Before applying the scheme

                }



                spdlog::info("Entering time stepping ADAPT");
                // tic();
                one_time_step_overleaves_corrected(f, pred_coeff, nb_ite);

       
                spdlog::info("Entering time stepping REFERENCE");
                one_time_step_overleaves_corrected(f_ref, pred_coeff, nb_ite);

            }
            
            auto error_density = compute_error(f, f_ref);
            std::cout<<std::endl<<"#### Epsilon = "<<eps<<"   error = "<<error_density<<std::flush;


        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
