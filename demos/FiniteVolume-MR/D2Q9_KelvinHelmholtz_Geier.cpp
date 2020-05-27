/*
    The choice of the momenti is that 
    of Geier as testesd in pyLBM and working properly
*/

#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <math.h> 

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


double mach   = 0.1;
double lambda = sqrt(3.0) / mach;
double rho_0  = 1.0;
double U_0    = 0.05;//0.5;
double zeta   = 0.0366;
double mu     = 1.0E-6;
double k      = 80.0;
double delta  = 0.05;


template<class coord_index_t>
auto compute_prediction_d2q9(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(8); // The null velocity is not used

        // Velocities parallel to the axis
        for (int l = 0; l < size; ++l)
        {
            data[k][0] += prediction(k, i*size - 1, j*size + l) - prediction(k, (i+1)*size - 1, j*size + l);
            data[k][1] += prediction(k, i*size + l, j*size - 1) - prediction(k, i*size + l, (j+1)*size - 1);
            data[k][2] += prediction(k, (i+1)*size, j*size + l) - prediction(k, i*size, j*size + l);
            data[k][3] += prediction(k, i*size + l, (j+1)*size) - prediction(k, i*size + l, j*size);
        }
        // Diagonal velocities -  x stripes
        for (int l = 0; l < size; ++l)
        {
            data[k][4] += prediction(k, i*size - l - 1, j*size - 1) - prediction(k, i*size + l, (j+1)*size - 1);
            data[k][5] += prediction(k, i*size + l + 1, j*size - 1) - prediction(k, i*size + l, (j+1)*size - 1);
            data[k][6] += prediction(k, i*size + l + 1, (j+1)*size) - prediction(k, i*size + l, j*size);
            data[k][7] += prediction(k, i*size + l - 1, (j+1)*size) - prediction(k, i*size + l, j*size);

        }
        // Diagonal velocities -  y stripes
        for (int l = 1; l < size; ++l) // We start from 1 in order not to count the angular cells twice
        {
            data[k][4] += prediction(k, i*size - 1, j*size + l - 1) - prediction(k, (i+1)*size - 1, j*size + l - 1);   
            data[k][5] += prediction(k, (i+1)*size, j*size + l - 1) - prediction(k, i*size, j*size + l - 1);   
            data[k][6] += prediction(k, (i+1)*size, j*size + l) - prediction(k, i*size, j*size + l);   
            data[k][7] += prediction(k, i*size - 1, j*size + l) - prediction(k, (i+1)*size - 1, j*size + l);   
        }

    }
    return data;
}


template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 9;
    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0}
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double rho = rho_0;
        double qx = 0.0;
        double qy = U_0 * delta * sin(2. * M_PI * (x + .25));

        if (y <= 0.5)  
            qx = U_0 * tanh(k * (y - .25));
        else
            qx = U_0 * tanh(k * (.75 - y));



        // We give standard names
        double cs2 = (lambda*lambda)/ 3.0; // sound velocity of the lattice squared

        double m0 = rho;
        double m1 = qx;
        double m2 = qy;
        double m3 = (qx*qx+qy*qy)/rho + 2.*rho*cs2;
        double m4 = qx*(cs2+(qy/rho)*(qy/rho));
        double m5 = qy*(cs2+(qx/rho)*(qx/rho));
        double m6 = rho*(cs2+(qx/rho)*(qx/rho))*(cs2+(qy/rho)*(qy/rho));
        double m7 = (qx*qx-qy*qy)/rho;
        double m8 = qx*qy/rho;

        // We come back to the distributions

        double r1 = 1.0 / lambda;
        double r2 = 1.0 / (lambda*lambda);
        double r3 = 1.0 / (lambda*lambda*lambda);
        double r4 = 1.0 / (lambda*lambda*lambda*lambda);

        f[cell][0] = m0                      -     r2*m3                        +     r4*m6                         ;
        f[cell][1] =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
        f[cell][2] =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
        f[cell][3] =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
        f[cell][4] =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
        f[cell][5] =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
        f[cell][6] =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
        f[cell][7] =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
        f[cell][8] =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;

    });

    return f;
}


template<class Field>
void one_time_step(Field &f)
{
    constexpr std::size_t nvel = Field::size;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    double space_step = 1.0 / (1 << max_level);

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);


    f.update_bc(); // Should we do it twice ?

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto& index, auto &interval, auto) {
            auto k = interval[0]; // Logical index in x
            auto h = index[0];    // Logical index in y

            // Uniform mesh for the moment

            auto f0 = xt::eval(f(0, level, k    , h    ));
            auto f1 = xt::eval(f(1, level, k - 1, h    ));
            auto f2 = xt::eval(f(2, level, k    , h - 1));
            auto f3 = xt::eval(f(3, level, k + 1, h    ));
            auto f4 = xt::eval(f(4, level, k    , h + 1));
            auto f5 = xt::eval(f(5, level, k - 1, h - 1));
            auto f6 = xt::eval(f(6, level, k + 1, h - 1));
            auto f7 = xt::eval(f(7, level, k + 1, h + 1));
            auto f8 = xt::eval(f(8, level, k - 1, h + 1));


            // // We compute the advected momenti
            double l1 = lambda;
            double l2 = l1 * lambda;
            double l3 = l2 * lambda;
            double l4 = l3 * lambda;

            auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
            auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
            auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
            auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
            auto m4 = xt::eval(l3*(                           f5 -   f6 -   f7 +   f8 ) );
            auto m5 = xt::eval(l3*(                           f5 +   f6 -   f7 -   f8 ) );
            auto m6 = xt::eval(l4*(                           f5 +   f6 +   f7 +   f8 ) );
            auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
            auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

            // Collision

            double dummy = 3.0/(lambda*rho_0*space_step);
            double sigma_1 = dummy*zeta;
            double sigma_2 = dummy*mu;
            double s_1 = 1/(.5+sigma_1);
            double s_2 = 1/(.5+sigma_2);

            double cs2 = (lambda * lambda) / 3.0; // sound velocity squared

            m3 = (1. - s_1) * m3 + s_1 * ((m1*m1+m2*m2)/m0 + 2.*m0*cs2);
            m4 = (1. - s_1) * m4 + s_1 * (m1*(cs2+(m2/m0)*(m2/m0)));
            m5 = (1. - s_1) * m5 + s_1 * (m2*(cs2+(m1/m0)*(m1/m0)));
            m6 = (1. - s_1) * m6 + s_1 * (m0*(cs2+(m1/m0)*(m1/m0))*(cs2+(m2/m0)*(m2/m0)));
            m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
            m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);


            // We come back to the distributions

            double r1 = 1.0 / lambda;
            double r2 = 1.0 / (lambda*lambda);
            double r3 = 1.0 / (lambda*lambda*lambda);
            double r4 = 1.0 / (lambda*lambda*lambda*lambda);


            new_f(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                         ;
            new_f(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            new_f(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            new_f(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            new_f(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            new_f(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            new_f(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
            new_f(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            new_f(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;


        });
    }

    std::swap(f.array(), new_f.array());
}


template<class Field, class Pred>
void one_time_step_overleaves(Field &f, const Pred & pred_coeff)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    double space_step = 1.0 / (1 << max_level);

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);
    mure::mr_prediction_overleaves(f);



    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    // This stored the fluxes computed at the level
    // of the overleaves
    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {



        if (level == max_level) {
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto f0 = xt::eval(f(0, level, k    , h    ));
                auto f1 = xt::eval(f(1, level, k - 1, h    ));
                auto f2 = xt::eval(f(2, level, k    , h - 1));
                auto f3 = xt::eval(f(3, level, k + 1, h    ));
                auto f4 = xt::eval(f(4, level, k    , h + 1));
                auto f5 = xt::eval(f(5, level, k - 1, h - 1));
                auto f6 = xt::eval(f(6, level, k + 1, h - 1));
                auto f7 = xt::eval(f(7, level, k + 1, h + 1));
                auto f8 = xt::eval(f(8, level, k - 1, h + 1));

                // // We compute the advected momenti
                double l1 = lambda;
                double l2 = l1 * lambda;
                double l3 = l2 * lambda;
                double l4 = l3 * lambda;

                auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
                auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
                auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
                auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
                auto m4 = xt::eval(l3*(                           f5 -   f6 -   f7 +   f8 ) );
                auto m5 = xt::eval(l3*(                           f5 +   f6 -   f7 -   f8 ) );
                auto m6 = xt::eval(l4*(                           f5 +   f6 +   f7 +   f8 ) );
                auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
                auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

                // Collision

                double dummy = 3.0/(lambda*rho_0*space_step);
                double sigma_1 = dummy*zeta;
                double sigma_2 = dummy*mu;
                double s_1 = 1/(.5+sigma_1);
                double s_2 = 1/(.5+sigma_2);

                double cs2 = (lambda * lambda) / 3.0; // sound velocity squared

                m3 = (1. - s_1) * m3 + s_1 * ((m1*m1+m2*m2)/m0 + 2.*m0*cs2);
                m4 = (1. - s_1) * m4 + s_1 * (m1*(cs2+(m2/m0)*(m2/m0)));
                m5 = (1. - s_1) * m5 + s_1 * (m2*(cs2+(m1/m0)*(m1/m0)));
                m6 = (1. - s_1) * m6 + s_1 * (m0*(cs2+(m1/m0)*(m1/m0))*(cs2+(m2/m0)*(m2/m0)));
                m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
                m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);


                // We come back to the distributions

                double r1 = 1.0 / lambda;
                double r2 = 1.0 / (lambda*lambda);
                double r3 = 1.0 / (lambda*lambda*lambda);
                double r4 = 1.0 / (lambda*lambda*lambda*lambda);


                new_f(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                           ;
                new_f(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
                new_f(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
            });

        }

        else
        {
            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1); 
            double coeff = 1. / (1 << (2*j)); // The 2 comes from the spatial dimension

            // We take the overleaves corresponding to the existing leaves
            auto overleaves = mure::intersection(mesh[mure::MeshType::all_cells][level], 
                                                mure::intersection(mesh[mure::MeshType::overleaves][level + 1],
                                                                   mesh[mure::MeshType::cells][level]))
                             .on(level + 1);

            
            overleaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


                //auto f0 = xt::eval(f(0, level, k    , h    ));
                auto f1 = xt::eval(0.0 * f(1, level, k - 1, h    )); // for the shape
                auto f2 = xt::eval(0.0 * f(2, level, k    , h - 1)); // for the shape
                auto f3 = xt::eval(0.0 * f(3, level, k + 1, h    )); // for the shape
                auto f4 = xt::eval(0.0 * f(4, level, k    , h + 1)); // for the shape
                auto f5 = xt::eval(0.0 * f(5, level, k - 1, h - 1)); // for the shape
                auto f6 = xt::eval(0.0 * f(6, level, k + 1, h - 1)); // for the shape
                auto f7 = xt::eval(0.0 * f(7, level, k + 1, h + 1)); // for the shape
                auto f8 = xt::eval(0.0 * f(8, level, k - 1, h + 1)); // for the shape


                // The velocity 0 is skept
                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f1 += coeff*c.second*f(1, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f2 += coeff*c.second*f(2, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][2].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f3 += coeff*c.second*f(3, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][3].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f4 += coeff*c.second*f(4, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][4].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f5 += coeff*c.second*f(5, level + 1, k + stencil_x, h + stencil_y);
                }
                
                for(auto &c: pred_coeff[j][5].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f6 += coeff*c.second*f(6, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][6].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f7 += coeff*c.second*f(7, level + 1, k + stencil_x, h + stencil_y);
                }

                for(auto &c: pred_coeff[j][7].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f8 += coeff*c.second*f(8, level + 1, k + stencil_x, h + stencil_y);
                }

                // // We save the fluxes
                // Probably not the most optimized choice...
                fluxes(1, level + 1, k, h) = f1;
                fluxes(2, level + 1, k, h) = f2;
                fluxes(3, level + 1, k, h) = f3;
                fluxes(4, level + 1, k, h) = f4;
                fluxes(5, level + 1, k, h) = f5;
                fluxes(6, level + 1, k, h) = f6;
                fluxes(7, level + 1, k, h) = f7;
                fluxes(8, level + 1, k, h) = f8;
            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto f0 = xt::eval(f(0, level, k, h));

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

                // // We compute the advected momenti
                double l1 = lambda;
                double l2 = l1 * lambda;
                double l3 = l2 * lambda;
                double l4 = l3 * lambda;

                auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
                auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
                auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
                auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
                auto m4 = xt::eval(l3*(                           f5 -   f6 -   f7 +   f8 ) );
                auto m5 = xt::eval(l3*(                           f5 +   f6 -   f7 -   f8 ) );
                auto m6 = xt::eval(l4*(                           f5 +   f6 +   f7 +   f8 ) );
                auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
                auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

                // Collision

                double dummy = 3.0/(lambda*rho_0*space_step);
                double sigma_1 = dummy*zeta;
                double sigma_2 = dummy*mu;
                double s_1 = 1/(.5+sigma_1);
                double s_2 = 1/(.5+sigma_2);

                double cs2 = (lambda * lambda) / 3.0; // sound velocity squared

                m3 = (1. - s_1) * m3 + s_1 * ((m1*m1+m2*m2)/m0 + 2.*m0*cs2);
                m4 = (1. - s_1) * m4 + s_1 * (m1*(cs2+(m2/m0)*(m2/m0)));
                m5 = (1. - s_1) * m5 + s_1 * (m2*(cs2+(m1/m0)*(m1/m0)));
                m6 = (1. - s_1) * m6 + s_1 * (m0*(cs2+(m1/m0)*(m1/m0))*(cs2+(m2/m0)*(m2/m0)));
                m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
                m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);


                // We come back to the distributions

                double r1 = 1.0 / lambda;
                double r2 = 1.0 / (lambda*lambda);
                double r3 = 1.0 / (lambda*lambda*lambda);
                double r4 = 1.0 / (lambda*lambda*lambda*lambda);


                new_f(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                         ;
                new_f(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7           ;
                new_f(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7           ;
                new_f(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7           ;
                new_f(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7           ;
                new_f(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
                new_f(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
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
    str << "LBM_D2Q9_KelvinHelmholtz_Geier_vort_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> rho{"rho", mesh};
    mure::Field<Config> qx{"qx", mesh};
    mure::Field<Config> qy{"qy", mesh};
    mure::Field<Config> vel_mod{"vel_modulus", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4] 
                               + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];

        qx[cell] = lambda * (f[cell][1] - f[cell][3] + f[cell][5] - f[cell][6] - f[cell][7] + f[cell][8]);
        qy[cell] = lambda * (f[cell][2] - f[cell][4] + f[cell][5] + f[cell][6] - f[cell][7] - f[cell][8]);

        vel_mod[cell] = xt::sqrt(qx[cell] * qx[cell] 
                               + qy[cell] * qy[cell]) / rho[cell];

    });

    mure::Field<Config> vort{"vorticity", mesh};

    // We update the ghosts
    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {

        double dx =  1.0 / (1 << (max_level - level));
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);


        exp([&](auto& index, auto &interval, auto) {
            auto k = interval[0]; // Logical index in x
            auto h = index[0];    // Logical index in y

            vort(level, k, h) = 1./(2.*dx) * ((lambda * (f(2, level, k + 1, h) - f(4, level, k + 1, h) 
                                                       + f(5, level, k + 1, h) + f(6, level, k + 1, h) - f(7, level, k + 1, h) - f(8, level, k + 1, h))) 
                                             / (f(0, level, k + 1, h) 
                                              + f(1, level, k + 1, h) + f(2, level, k + 1, h) + f(3, level, k + 1, h) + f(4, level, k + 1, h) 
                                              + f(5, level, k + 1, h) + f(6, level, k + 1, h) + f(7, level, k + 1, h) + f(8, level, k + 1, h))) 
                              - 1./(2.*dx) * ((lambda * (f(2, level, k - 1, h) - f(4, level, k - 1, h) 
                                                       + f(5, level, k - 1, h) + f(6, level, k - 1, h) - f(7, level, k - 1, h) - f(8, level, k - 1, h))) 
                                             / (f(0, level, k - 1, h) 
                                              + f(1, level, k - 1, h) + f(2, level, k - 1, h) + f(3, level, k - 1, h) + f(4, level, k - 1, h) 
                                              + f(5, level, k - 1, h) + f(6, level, k - 1, h) + f(7, level, k - 1, h) + f(8, level, k - 1, h)))
                              + 1./(2.*dx) * ((lambda * (f(1, level, k, h - 1) - f(3, level, k, h - 1) 
                                                       + f(5, level, k, h - 1) - f(6, level, k, h - 1) - f(7, level, k, h - 1) + f(8, level, k, h - 1))) 
                                             / (f(0, level, k, h - 1) 
                                              + f(1, level, k, h - 1) + f(2, level, k, h - 1) + f(3, level, k, h - 1) + f(4, level, k, h - 1) 
                                              + f(5, level, k, h - 1) + f(6, level, k, h - 1) + f(7, level, k, h - 1) + f(8, level, k, h - 1)))
                              - 1./(2.*dx) * ((lambda * (f(1, level, k, h + 1) - f(3, level, k, h + 1) 
                                                       + f(5, level, k, h + 1) - f(6, level, k, h + 1) - f(7, level, k, h + 1) + f(8, level, k, h + 1))) 
                                             / (f(0, level, k, h + 1) 
                                              + f(1, level, k, h + 1) + f(2, level, k, h + 1) + f(3, level, k, h + 1) + f(4, level, k, h + 1) 
                                              + f(5, level, k, h + 1) + f(6, level, k, h + 1) + f(7, level, k, h + 1) + f(8, level, k, h + 1)));                  

        });

    }

    h5file.add_field(rho);
    h5file.add_field(qx);
    h5file.add_field(qy);
    h5file.add_field(vel_mod);

    h5file.add_field(f);
    h5file.add_field(level_);

    h5file.add_field(vort);

}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q5_kelvin_helhotlz",
                             "...");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("8"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("8"))
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
            auto pred_coeff = compute_prediction_d2q9<coord_index_t>(min_level, max_level);

            // Initialization
            auto f = init_f(mesh, 0);

            double T = 20.0;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < 1; ++nb_ite)
            {
                std::cout <<"Iteration" << nb_ite<<" Time = "<<nb_ite * dt << "\n";



                if (nb_ite % 50 == 0)
                    save_solution(f, eps, nb_ite / 50);


                //save_solution(f, eps, nb_ite);

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (coarsening(f, eps, i))
                        break;
                }
                // std::cout << "coarsening\n";
                // // save_solution(f, eps, nb_ite, "coarsening");

                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (refinement(f, eps, 1.0, i))
                        break;
                }
                // std::cout << "refinement\n";

                f.update_bc();

                // if (nb_ite % 50 == 0) {
                //     std::stringstream str;
                //     str << "debug_by_level_"<<nb_ite;

                //     auto h5file = mure::Hdf5(str.str().data());
                //     h5file.add_field_by_level(mesh, f);

                // }

                //save_solution(f, eps, nb_ite, "refinement");

                //one_time_step(f);
                one_time_step_overleaves(f, pred_coeff);

                // save_solution(f, eps, nb_ite);
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
