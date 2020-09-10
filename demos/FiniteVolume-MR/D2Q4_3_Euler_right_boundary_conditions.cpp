#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"

// double lambda = 1./0.3; //4.0; // We always used it
double lambda = 1./0.2499; //4.0;

double sigma_q = 0.5; 
double sigma_xy = 0.5;

double sq = 1.5;//1./(.5 + sigma_q);
double sxy = 1./(.5 + sigma_xy);

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
        
        double gm = 1.4;

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

                // rho = 0.8;
                // qx = rho * 0.1;
                // qy = rho * 0.0;
                // p = 0.4;

                rho = 1.0625;
                qx = rho * 0.;
                qy = rho * 0.2145;
                p = 0.4;

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

                // rho = 0.5313;
                // qx = rho * 0.8276;
                // qy = rho * 0.0;
                // p = 0.4;

                rho = 2.;
                qx = rho * 0.;
                qy = rho * (-0.3);
                p = 1.;
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

                // rho = 0.5313;
                // qx = rho * 0.1;
                // qy = rho * 0.7276;
                // p = 0.4;

                rho = 0.5197;
                qx = rho * 0.;
                qy = rho * (-1.1259);
                p = 0.4;
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

                                
                // rho = 1.;
                // qx = rho * 0.1;
                // qy = rho * 0.0;
                // p = 1.;

                rho = 1.;
                qx = rho * 0.;
                qy = rho * (-0.4);
                p = 1.;
            }
        }
        
        
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


        data[k].resize(8);


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

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp)),
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
    auto max_level = mesh.max_level();
    
    std::cout<<std::endl<<"[+] Projecting"<<std::flush;
    mure::mr_projection(f);
    std::cout<<std::endl<<"[+] Updating BC"<<std::flush;
    f.update_bc(); // It is important to do so
    std::cout<<std::endl<<"[+] Predicting"<<std::flush;
    mure::mr_prediction(f);
    std::cout<<std::endl<<"[+] Predicting overleaves"<<std::flush;
    mure::mr_prediction_overleaves(f);

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    // This stored the fluxes computed at the level
    // of the overleaves
    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);


    Field advected{"advected", mesh};
    advected.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        if (level == max_level) {

            std::cout<<std::endl<<"[+] Advecting at finest"<<std::flush;
            
            std::cout<<std::endl<<"[=] East"<<std::flush;
            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, mure::MeshType::cells);
            leaves_east.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a flat BC
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); // Direct evaluation of the BC
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
            
            std::cout<<std::endl<<"[=] North"<<std::flush;
            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mure::MeshType::cells);
            leaves_north.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a flat BC
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1 ,h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mure::MeshType::cells);
            leaves_northeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mure::MeshType::cells);
            leaves_west.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mure::MeshType::cells);
            leaves_northwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });

            std::cout<<std::endl<<"[=] South"<<std::flush;
            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mure::MeshType::cells);
            leaves_south.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mure::MeshType::cells);
            leaves_southwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mure::MeshType::cells);
            leaves_southeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            // Advection far from the boundary
            auto tmp1 = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2 = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);
            auto internal_leaves = mure::difference(mesh[mure::MeshType::cells][max_level],
                                      all_leaves_boundary).on(max_level); // It is very important to project at this point

            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            internal_leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            // Its time for collision which is local
            auto leaves = intersection(mesh[mure::MeshType::cells][max_level], 
                                       mesh[mure::MeshType::cells][max_level]);

            std::cout<<std::endl<<"[+] Colliding at finest"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y  

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

                double gm = 1.4;

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

        else
        {

            std::size_t j = max_level - (level + 1);
            double coeff = 1. / (1 << (2*j)); // ATTENTION A LA DIMENSION 2 !!!!

            std::cout<<std::endl<<"[+] Advecting at level "<<level<<" with overleaves at "<<(level + 1)<<std::flush;

            // This is necessary because the only overleaves we have to advect
            // on are the ones superposed with the leaves to which we come back
            // eventually in the process
            auto overleaves_east = intersection(get_adjacent_boundary_east(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northeast = intersection(get_adjacent_boundary_northeast(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southeast = intersection(get_adjacent_boundary_southeast(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_east = union_(union_(overleaves_east, overleaves_northeast), 
                                        overleaves_southeast);

            // General
            std::cout<<std::endl<<"[=] East/NorthEast/SouthEast"<<std::flush;
            touching_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    
                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });
            // Corrections
            std::cout<<std::endl<<"[=] East"<<std::flush;
            overleaves_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(2 + 4 * scheme_n, level + 1, k, h);
                }
            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            overleaves_northeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(3 + 4 * scheme_n, level + 1, k, h);
                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(2 + 4 * scheme_n, level + 1, k, h);
                }
            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            overleaves_southeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {

                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(1 + 4 * scheme_n, level + 1, k, h);
                    
                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(2 + 4 * scheme_n, level + 1, k, h);
                }
                
            });


            auto overleaves_west = intersection(get_adjacent_boundary_west(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northwest = intersection(get_adjacent_boundary_northwest(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southwest = intersection(get_adjacent_boundary_southwest(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_west = union_(union_(overleaves_west, overleaves_northwest), 
                                        overleaves_southwest);

            std::cout<<std::endl<<"[=] West/NorthWest/SouthWest"<<std::flush;
            touching_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            overleaves_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(0 + 4 * scheme_n, level + 1, k, h);

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            overleaves_northwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(0 + 4 * scheme_n, level + 1, k, h);

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(3 + 4 * scheme_n, level + 1, k, h);
                    
                }
            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            overleaves_southwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(0 + 4 * scheme_n, level + 1, k, h);

                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(1 + 4 * scheme_n, level + 1, k, h);
                    

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });


            auto overleaves_south = intersection(get_adjacent_boundary_south(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_north = intersection(get_adjacent_boundary_north(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
                                                                                    
            auto north_and_south = union_(overleaves_south, overleaves_north);

            std::cout<<std::endl<<"[=] North/South"<<std::flush;
            north_and_south.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

            });
                                    

            std::cout<<std::endl<<"[=] South"<<std::flush;
            overleaves_south.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {

                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(1 + 4 * scheme_n, level + 1, k, h);
                
  
                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }                    
                }

            });

            std::cout<<std::endl<<"[=] North"<<std::flush;
            overleaves_north.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

  
                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * f(3 + 4 * scheme_n, level + 1, k, h);
                    
                }

            });
                    


            // // To update
            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            auto overleaves_far_boundary = difference(mesh[mure::MeshType::cells][level], 
                                                      union_(union_(touching_east, touching_west), 
                                                             north_and_south)).on(level + 1);  // Again, it is very important to project before using

            overleaves_far_boundary([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

            });

// Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            std::cout<<std::endl<<"[+] Projection of the overleaves on their leaves and collision"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto f0 = xt::eval(f(0, level, k, h)) + 0.25 * (fluxes(0, level + 1, 2*k,     2*h) 
                                                              + fluxes(0, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(0, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(0, level + 1, 2*k + 1, 2*h + 1));

                auto f1 = xt::eval(f(1, level, k, h)) + 0.25 * (fluxes(1, level + 1, 2*k,     2*h) 
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(1, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h + 1));

                auto f2 = xt::eval(f(2, level, k, h)) + 0.25 * (fluxes(2, level + 1, 2*k,     2*h) 
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(2, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h + 1));

                auto f3 = xt::eval(f(3, level, k, h)) + 0.25 * (fluxes(3, level + 1, 2*k,     2*h) 
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(3, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h + 1));


                auto f4 = xt::eval(f(4, level, k, h)) + 0.25 * (fluxes(4, level + 1, 2*k,     2*h) 
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(4, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h + 1));

                auto f5 = xt::eval(f(5, level, k, h)) + 0.25 * (fluxes(5, level + 1, 2*k,     2*h) 
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(5, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h + 1));

                auto f6 = xt::eval(f(6, level, k, h)) + 0.25 * (fluxes(6, level + 1, 2*k,     2*h) 
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(6, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h + 1));

                auto f7 = xt::eval(f(7, level, k, h)) + 0.25 * (fluxes(7, level + 1, 2*k,     2*h) 
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(7, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h + 1));


                auto f8 = xt::eval(f(8, level, k, h)) + 0.25 * (fluxes(8, level + 1, 2*k,     2*h) 
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(8, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h + 1));

                auto f9 = xt::eval(f(9, level, k, h)) + 0.25 * (fluxes(9, level + 1, 2*k,     2*h) 
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(9, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h + 1));

                auto f10 = xt::eval(f(10, level, k, h)) + 0.25 * (fluxes(10, level + 1, 2*k,     2*h) 
                                                                + fluxes(10, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(10, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(10, level + 1, 2*k + 1, 2*h + 1));

                auto f11 = xt::eval(f(11, level, k, h)) + 0.25 * (fluxes(11, level + 1, 2*k,     2*h) 
                                                                + fluxes(11, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(11, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(11, level + 1, 2*k + 1, 2*h + 1));


                auto f12 = xt::eval(f(12, level, k, h)) + 0.25 * (fluxes(12, level + 1, 2*k,     2*h) 
                                                                + fluxes(12, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(12, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(12, level + 1, 2*k + 1, 2*h + 1));

                auto f13 = xt::eval(f(13, level, k, h)) + 0.25 * (fluxes(13, level + 1, 2*k,     2*h) 
                                                                + fluxes(13, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(13, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(13, level + 1, 2*k + 1, 2*h + 1));

                auto f14 = xt::eval(f(14, level, k, h)) + 0.25 * (fluxes(14, level + 1, 2*k,     2*h) 
                                                                + fluxes(14, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(14, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(14, level + 1, 2*k + 1, 2*h + 1));

                auto f15 = xt::eval(f(15, level, k, h)) + 0.25 * (fluxes(15, level + 1, 2*k,     2*h) 
                                                                + fluxes(15, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(15, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(15, level + 1, 2*k + 1, 2*h + 1));

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


                double gm = 1.4;

                // m0_1 = (1 - sq) *  m0_1 + sq * (m1_0);
                // m0_2 = (1 - sq) *  m0_2 + sq * (m2_0);
                // m0_3 = (1 - sxy) * m0_3; 


                // m1_1 = (1 - sq) *  m1_1 + sq * ((3./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (1./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (gm - 1.) * m3_0);
                // m1_2 = (1 - sq) *  m1_2 + sq * (m1_0*m2_0/m0_0);
                // m1_3 = (1 - sxy) * m1_3; 

                // m2_1 = (1 - sq) *  m2_1 + sq * (m1_0*m2_0/m0_0);
                // m2_2 = (1 - sq) *  m2_2 + sq * ((3./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (1./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (gm - 1.) * m3_0);
                // m2_3 = (1 - sxy) * m2_3; 

                // m3_1 = (1 - sq) *  m3_1 + sq * (gm*(m1_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m1_0*m1_0*m1_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m1_0*m2_0*m2_0)/(m0_0*m0_0));
                // m3_2 = (1 - sq) *  m3_2 + sq * (gm*(m2_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m2_0*m2_0*m2_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m2_0*m1_0*m1_0)/(m0_0*m0_0));
                // m3_3 = (1 - sxy) * m3_3; 


                std::size_t how_often = 1 << (max_level - level);

                double sq_real  = (iter % how_often == 0) ? sq  : 0.;
                double sxy_real = (iter % how_often == 0) ? sxy : 0.;

                m0_1 = (1 - sq_real) *  m0_1 + sq_real * (m1_0);
                m0_2 = (1 - sq_real) *  m0_2 + sq_real * (m2_0);
                m0_3 = (1 - sxy_real) * m0_3; 


                m1_1 = (1 - sq_real) *  m1_1 + sq_real * ((3./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (1./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (gm - 1.) * m3_0);
                m1_2 = (1 - sq_real) *  m1_2 + sq_real * (m1_0*m2_0/m0_0);
                m1_3 = (1 - sxy_real) * m1_3; 

                m2_1 = (1 - sq_real) *  m2_1 + sq_real * (m1_0*m2_0/m0_0);
                m2_2 = (1 - sq_real) *  m2_2 + sq_real * ((3./2. - gm/2.) * (m2_0*m2_0)/(m0_0) + (1./2. - gm/2.) * (m1_0*m1_0)/(m0_0) + (gm - 1.) * m3_0);
                m2_3 = (1 - sxy_real) * m2_3; 

                m3_1 = (1 - sq_real) *  m3_1 + sq_real * (gm*(m1_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m1_0*m1_0*m1_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m1_0*m2_0*m2_0)/(m0_0*m0_0));
                m3_2 = (1 - sq_real) *  m3_2 + sq_real * (gm*(m2_0*m3_0)/(m0_0) + (gm/2. - 1./2.)*(m2_0*m2_0*m2_0)/(m0_0*m0_0) + + (gm/2. - 1./2.)*(m2_0*m1_0*m1_0)/(m0_0*m0_0));
                m3_3 = (1 - sxy_real) * m3_3; 


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
    str << "LBM_D2Q4_3_Euler_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> rho{"rho", mesh};
    mure::Field<Config> qx{"qx", mesh};
    mure::Field<Config> qy{"qy", mesh};
    mure::Field<Config> e{"e", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
        qx[cell]  = f[cell][4] + f[cell][5] + f[cell][6] + f[cell][7];
        qy[cell]  = f[cell][8] + f[cell][9] + f[cell][10] + f[cell][11];
        e[cell]   = f[cell][12] + f[cell][13] + f[cell][14] + f[cell][15];

    });
    h5file.add_field(rho);
    h5file.add_field(qx);
    h5file.add_field(qy);
    h5file.add_field(e);

    h5file.add_field(f);
    h5file.add_field(level_);
}




int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Euler",
                             "Multi resolution for a D2Q4 LBM scheme for the scalar advection equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("7"))
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

            //auto save_string = std::string("bruteforce");
            auto save_string = std::string("overleaves");


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



            // std::cout<<std::endl<<"Showing prediction matrix for fluxes"<<std::endl;

            // for (int idx = 0; idx <= 7; ++idx){
            // std::cout<<std::endl<<"Idx = "<<idx<<std::endl;
            // for (int k = 0; k <= max_level - min_level; ++k)
            // {
            //     for (auto cf : pred_coeff[k][idx].coeff){
            //         coord_index_t stencil_x, stencil_y;
            //         std::tie(stencil_x, stencil_y) = cf.first;
                    
                    
            //         std::cout<<"k = "<<k<<"  Offset x = "<<stencil_x<<"   Offset y = "<<stencil_y<<"   Value = "<<cf.second<<std::endl;
            //     }
                   
            // }
            // }
            // return 0;

            // Initialization
            auto f = init_f(mesh, 0);

            double T = 0.3;//1.2;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout<<std::endl<<"Iteration number = "<<nb_ite<<std::endl;

                std::cout<<std::endl<<"[*] Coarsening"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (coarsening(f, eps, i))
                        break;
                }

                std::cout<<std::endl<<"[*] Refinement"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (refinement(f, eps, 0.0, i))
                        break;
                }

                std::cout<<std::endl<<"[*] Prediction overleaves before saving"<<std::flush;
                mure::mr_prediction_overleaves(f); // Before saving



                // if (nb_ite > 0)
                //save_solution(f, eps, nb_ite, std::string("fullcomp"));
                //save_solution(f, eps, nb_ite, std::string("nocorr"));
                //save_solution(f, eps, nb_ite);
                std::cout<<std::endl<<"[*] Saving solution"<<std::flush;
                save_solution(f, eps, nb_ite, save_string+std::string("_before")); // Before applying the scheme
       
                //save_solution(f, eps, nb_ite,std::string("bback") );


                // std::cout<<std::endl<<"Printing mesh "<<std::endl<<f.mesh()<<std::endl;
                // if (nb_ite < 30)    {
                //     f.update_bc();
                //     std::stringstream str;
                //     str << "debug_by_level_"<<save_string<<"_before_"<<nb_ite;

                //     auto h5file = mure::Hdf5(str.str().data());
                //     h5file.add_field_by_level(mesh, f);
                // }
                //return 0;

                //one_time_step_with_mem(f, nb_ite);
                //one_time_step(f,pred_coeff);
                //one_time_step_overleaves(f, pred_coeff);
                std::cout<<std::endl<<"[*] Entering time stepping"<<std::flush;
                one_time_step_overleaves_corrected(f, pred_coeff, nb_ite);


                // save_solution(f, eps, nb_ite, save_string+std::string("_after")); // Before applying the scheme
                // if (nb_ite < 30)    {
                //     std::stringstream str;
                //     str << "debug_by_level_"<<save_string<<"_after_"<<nb_ite;

                //     auto h5file = mure::Hdf5(str.str().data());
                //     h5file.add_field_by_level(mesh, f);
                // }

            }
            
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
