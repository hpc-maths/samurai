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

double lambda = 2.;
double sigma_q = 0.5; 
double sigma_xy = 0.5;

double sq = 1.;//1./(.5 + sigma_q);
double sxy = 1./(.5 + sigma_xy);

double kx = sqrt(2.) / 2.0;
double ky = sqrt(2.) / 2.0;



template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 4;
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

        double m0 = 0.0;

        double radius = .1;
        double x_center = 0.5, y_center = 0.5;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                <= radius * radius)
            m0 = 1.;
        // if (abs(x - x_center) <= radius and abs(y - y_center) <= radius)    {
        //     m0 = 1.;
        // }


        // double m1 = 0.5 * kx * m0 * m0;
        // double m2 = 0.5 * ky * m0 * m0;


        double m1 = kx * m0;
        double m2 = ky * m0;
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
    f.update_bc(); // It is important to do so
    mure::mr_prediction(f);
    mure::mr_prediction_overleaves(f);

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

        if (level == max_level) {
            spdlog::info("Finest level treatment");

            sw.reset();

            spdlog::stopwatch sw_boundary;

            sw_boundary.reset();

            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, mure::MeshType::cells);
            leaves_east.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a flat BC
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); // Direct evaluation of the BC
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
            spdlog::info("COMPUTATION ADJACENT EAST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();

            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mure::MeshType::cells);
            leaves_north.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a flat BC
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1 ,h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });
            spdlog::info("COMPUTATION ADJACENT NORTH FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();

            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mure::MeshType::cells);
            leaves_northeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });
            spdlog::info("COMPUTATION ADJACENT NORTH EAST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();

            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mure::MeshType::cells);
            leaves_west.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
            spdlog::info("COMPUTATION ADJACENT WEST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();

            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mure::MeshType::cells);
            leaves_northwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });
            spdlog::info("COMPUTATION ADJACENT NORTH WEST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();


            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mure::MeshType::cells);
            leaves_south.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
            spdlog::info("COMPUTATION ADJACENT SOUTH FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();


            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mure::MeshType::cells);
            leaves_southwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
            
            spdlog::info("COMPUTATION ADJACENT SOUTH WEST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();


            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mure::MeshType::cells);
            leaves_southeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            spdlog::info("COMPUTATION ADJACENT SOUTH EAST FINEST = {:.3}", sw_boundary);
            sw_boundary.reset();

            spdlog::info("Advection boundary finest level = {:.3} s", sw);
            sw.reset();

            // Advection far from the boundary
            auto tmp1 = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2 = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);
            auto internal_leaves = mure::difference(mesh[mure::MeshType::cells][max_level],
                                      all_leaves_boundary).on(max_level); // It is very important to project at this point

            internal_leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            spdlog::info("Advection inside finest level = {:.3}", sw);
            sw.reset();

            // Its time for collision which is local
            auto leaves = intersection(mesh[mure::MeshType::cells][max_level], 
                                       mesh[mure::MeshType::cells][max_level]);

            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y  

                auto f0 = xt::eval(advected(0, level, k, h));
                auto f1 = xt::eval(advected(1, level, k, h));
                auto f2 = xt::eval(advected(2, level, k, h));
                auto f3 = xt::eval(advected(3, level, k, h));
                


                // We compute the advected momenti
                auto m0_0 = xt::eval(                 f0 + f1 + f2 + f3) ;
                auto m0_1 = xt::eval(lambda        * (f0      - f2      ));
                auto m0_2 = xt::eval(lambda        * (     f1      - f3));
                auto m0_3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));



                // m0_1 = (1. - sq) *  m0_1 + sq * 0.5 * kx * m0_0 * m0_0;
                // m0_2 = (1. - sq) *  m0_2 + sq * 0.5 * ky * m0_0 * m0_0;
                // m0_3 = (1. - sxy) * m0_3; 

                m0_1 = (1. - sq) *  m0_1 + sq * kx * m0_0;
                m0_2 = (1. - sq) *  m0_2 + sq * ky * m0_0;
                m0_3 = (1. - sxy) * m0_3; 


                new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;



            });

            spdlog::info("Collision finest level = {:.3}", sw);
        }
        else
        {
            auto lev_p_1 = level + 1;
            std::size_t j = max_level - (lev_p_1);

            spdlog::info("Overleaves treatment");
            sw.reset();
            // This is necessary because the only overleaves we have to advect
            // on are the ones superposed with the leaves to which we come back
            // eventually in the process
            auto overleaves_east = intersection(get_adjacent_boundary_east(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northeast = intersection(get_adjacent_boundary_northeast(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southeast = intersection(get_adjacent_boundary_southeast(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_east = union_(union_(overleaves_east, overleaves_northeast), 
                                        overleaves_southeast);

            // General
            touching_east.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    
                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(0 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(0 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(2 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                }
            });
            // Corrections
            overleaves_east.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(2 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(2 + 4 * scheme_n, lev_p_1, k, h);
                }
            });

            overleaves_northeast.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(3 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(3 + 4 * scheme_n, lev_p_1, k, h);
                    fluxes(2 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(2 + 4 * scheme_n, lev_p_1, k, h);
                }
            });

            overleaves_southeast.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {

                    fluxes(1 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(1 + 4 * scheme_n, lev_p_1, k, h);
                    
                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(2 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(2 + 4 * scheme_n, lev_p_1, k, h);
                }
                
            });


            auto overleaves_west = intersection(get_adjacent_boundary_west(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northwest = intersection(get_adjacent_boundary_northwest(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southwest = intersection(get_adjacent_boundary_southwest(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_west = union_(union_(overleaves_west, overleaves_northwest), 
                                        overleaves_southwest);

            touching_west.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(0 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(2 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(2 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            overleaves_west.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(0 + 4 * scheme_n, lev_p_1, k, h);

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            overleaves_northwest.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(0 + 4 * scheme_n, lev_p_1, k, h);

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    fluxes(3 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(3 + 4 * scheme_n, lev_p_1, k, h);
                    
                }
            });

            overleaves_southwest.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(0 + 4 * scheme_n, lev_p_1, k, h);

                    fluxes(1 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(1 + 4 * scheme_n, lev_p_1, k, h);
                    

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                }
            });


            auto overleaves_south = intersection(get_adjacent_boundary_south(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_north = intersection(get_adjacent_boundary_north(mesh, lev_p_1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
                                                                                    
            auto north_and_south = union_(overleaves_south, overleaves_north);

            north_and_south.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(0 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(0 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(2 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(2 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    
                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) -=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                }

            });
                                    

            overleaves_south.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {

                    fluxes(1 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(1 + 4 * scheme_n, lev_p_1, k, h);
                
  
                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(3 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }                    
                }

            });

            overleaves_north.on(lev_p_1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, lev_p_1, k, h) +=  c.second * f(1 + 4 * scheme_n, lev_p_1, k + stencil_x, h + stencil_y);
                    }

  
                    fluxes(3 + 4 * scheme_n, lev_p_1, k, h) += (1<<j) *  f(3 + 4 * scheme_n, lev_p_1, k, h);
                    
                }

            });

            // time_advection_overleaves_boundary += toc();
                    
            // tic();


            time_advection_overleaves_boundary += sw.elapsed().count();
            sw.reset();


            // // To update
            auto overleaves_far_boundary = difference(mesh[mure::MeshType::cells][level], 
                                                      union_(union_(touching_east, touching_west), 
                                                             north_and_south)).on(lev_p_1);  // Again, it is very important to project before using

            overleaves_far_boundary([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    {

                    auto shift = 4 * scheme_n;

                    for(auto &c: pred_coeff[j][8].coeff) 
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + shift, lev_p_1, k, h) +=  c.second * f(0 + shift, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                   
                    for(auto &c: pred_coeff[j][9].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + shift, lev_p_1, k, h) +=  c.second * f(1 + shift, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    

                    for(auto &c: pred_coeff[j][10].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + shift, lev_p_1, k, h) +=  c.second * f(2 + shift, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                    

                    for(auto &c: pred_coeff[j][11].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + shift, lev_p_1, k, h) +=  c.second * f(3 + shift, lev_p_1, k + stencil_x, h + stencil_y);
                    }
                  
                }

            });


            time_advection_overleaves_inside += sw.elapsed().count();
            sw.reset();

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto two_k = 2*k;
                auto two_k_p_1 = 2*k+1;
                auto two_h = 2*h;
                auto two_h_p_1 = 2*h+1;

                double coeff = 1. / (1 << (2*j)); // ATTENTION A LA DIMENSION 2 !!!!


                auto f0 = xt::eval(f(0, level, k, h) + coeff * 0.25 * (fluxes(0, lev_p_1, two_k,     two_h) 
                                                                     + fluxes(0, lev_p_1, two_k + 1, two_h)
                                                                     + fluxes(0, lev_p_1, two_k,     two_h_p_1)
                                                                     + fluxes(0, lev_p_1, two_k_p_1, two_h_p_1)));

                auto f1 = xt::eval(f(1, level, k, h) + coeff * 0.25 * (fluxes(1, lev_p_1, two_k,     two_h) 
                                                              + fluxes(1, lev_p_1, two_k_p_1, two_h)
                                                              + fluxes(1, lev_p_1, two_k,     two_h_p_1)
                                                              + fluxes(1, lev_p_1, two_k_p_1, two_h_p_1)));

                auto f2 = xt::eval(f(2, level, k, h) + coeff * 0.25 * (fluxes(2, lev_p_1, two_k,     two_h) 
                                                              + fluxes(2, lev_p_1, two_k_p_1, two_h)
                                                              + fluxes(2, lev_p_1, two_k,     two_h_p_1)
                                                              + fluxes(2, lev_p_1, two_k_p_1, two_h_p_1)));

                auto f3 = xt::eval(f(3, level, k, h) + coeff * 0.25 * (fluxes(3, lev_p_1, two_k,     two_h) 
                                                              + fluxes(3, lev_p_1, two_k_p_1, two_h)
                                                              + fluxes(3, lev_p_1, two_k,     two_h_p_1)
                                                              + fluxes(3, lev_p_1, two_k_p_1, two_h_p_1)));



                
                // We compute the advected momenti
                auto m0_0 = xt::eval(                 f0 + f1 + f2 + f3) ;
                auto m0_1 = xt::eval(lambda        * (f0      - f2      ));
                auto m0_2 = xt::eval(lambda        * (     f1      - f3));
                auto m0_3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));






                // m0_1 = (1. - sq) *  m0_1 + sq * 0.5 * kx * m0_0 * m0_0;
                // m0_2 = (1. - sq) *  m0_2 + sq * 0.5 * ky * m0_0 * m0_0;
                // m0_3 = (1. - sxy) * m0_3; 


                m0_1 = (1. - sq) *  m0_1 + sq * kx * m0_0;
                m0_2 = (1. - sq) *  m0_2 + sq * ky * m0_0;
                m0_3 = (1. - sxy) * m0_3; 


                new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;


            });

            time_collision_overleaves += sw.elapsed().count();
        }
    }

    spdlog::info("Advection overleaves boundary = {:.3} s", time_advection_overleaves_boundary);
    spdlog::info("Advection overleaves inside = {:.3} s", time_advection_overleaves_inside);
    spdlog::info("Collision overleaves = {:.3} s", time_collision_overleaves);


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
    str << "LBM_D2Q4_Adevction_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> rho{"rho", mesh};



    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];


    });
    h5file.add_field(rho);


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
    std::vector<std::size_t> shape_x = {k.size(), 4};
    xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(mure::MeshType::cells_and_ghosts, level_g + level, k, h); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);
        
    for (int h_field = 0; h_field < 4; ++h_field)  {
        xt::view(mask_all, xt::all(), h_field) = mask;
    }    

    // Recursion finished
    if (xt::all(mask))
    {                 
        return xt::eval(f(0, 4, level_g + level, k, h));
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
            xt::view(out, k_mask) = xt::view(f(0, 4, level_g + level, {k_int, k_int + 1}, h), 0);
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

  
    mure::Field<Config, double, 4> f_reconstructed("f_reconstructed", init_mesh, bc);
    f_reconstructed.array().fill(0.);

    mure::Field<Config> rho_reconstructed{"rho_reconstructed", init_mesh};
    mure::Field<Config> rho{"rho", init_mesh};


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

            rho(max_level, k, h) = f_full(0, max_level, k, h)
                                 + f_full(1, max_level, k, h)
                                 + f_full(2, max_level, k, h)
                                 + f_full(3, max_level, k, h);

        });
    }


    std::cout<<std::endl;

    std::stringstream str;
    str << "Advection_Reconstruction_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(init_mesh);
    h5file.add_field(rho_reconstructed);
    h5file.add_field(rho);

}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_",
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
            mure::Mesh<Config> mesh_ref{box, max_level, max_level};

            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

            // Initialization
            auto f = init_f(mesh, 0);
            auto f_ref = init_f(mesh_ref, 0);

            double T = 0.5;
            double dx = 1.0 / (1 << max_level);
            double dt = dx;


            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout << nb_ite << "\n";


                // for (std::size_t i=0; i<max_level-min_level; ++i)
                // {
                //     if (coarsening(f, eps, i))
                //         break;
                // }


                // for (std::size_t i=0; i<max_level-min_level; ++i)
                // {
                //     if (refinement(f, eps, 0.0, i))
                //         break;
                // }

                bool make_graduation = true;

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (harten(f, eps, 2., i, make_graduation))
                        break;
                }


                make_graduation = false;

                mure::mr_prediction_overleaves(f); // Before saving

                // if (nb_ite == N)    {
                //     save_reconstructed(f, f_ref, eps, 0);
                //     save_solution(f, eps, 0, save_string+std::string("PAPER")); // Before applying the scheme

                // }


                save_solution(f, eps, nb_ite, save_string+std::string("_before")); // Before applying the scheme

                one_time_step_overleaves_corrected(f, pred_coeff, nb_ite);
                one_time_step_overleaves_corrected(f_ref, pred_coeff, nb_ite);



            }
            
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
