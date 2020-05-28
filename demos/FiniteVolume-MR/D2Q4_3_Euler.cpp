#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"

double lambda = 4.0;
double sigma_q = 0.5; 
double sigma_xy = 0.5;

double sq = 1.0;//1./(.5 + sigma_q);
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

        double p = 0.0;
        
        double gm = 1.4;

        if (x < 0.5)    {
            if (y < 0.5)    {
                // 3
                // rho = 1.0;
                // qx = rho * 0.75;
                // qy = rho * 0.5;
                // double p = 1.0;

                rho = 0.5;
                qx = rho * 0.0;
                qy = rho * 0.0;
                p = 1.0;

            }
            else
            {
                // 2   
                // rho = 2.0;
                // qx = rho * (-0.75);
                // qy = rho * (0.5);
                // double p = 1.0;

                rho = 0.5;
                qx = rho * 0.7276;
                qy = rho * 0.0;
                p = 1.0;
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

                rho = 1.0;
                qx = rho * 0.0;
                qy = rho * 0.7276;
                p = 1.0;
            }
            else
            {
                // 1
                // rho = 1.0;
                // qx = rho * (-0.75);
                // qy = rho * (-0.5);
                // double p = 1.0;
                
                rho = 0.5313;
                qx = rho * 0.0;
                qy = rho * 0.0;
                p = 0.4;
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

// We have to average only the fluxes
template<class Field, class pred>
void one_time_step_overleaves_corrected(Field &f, const pred& pred_coeff, std::size_t iter)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
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

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        if (level == max_level) {
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto f0 = xt::eval(f(0, level, k - 1, h    ));
                auto f1 = xt::eval(f(1, level, k,     h - 1));
                auto f2 = xt::eval(f(2, level, k + 1, h    ));
                auto f3 = xt::eval(f(3, level, k,     h + 1));

                auto f4 = xt::eval(f(4, level, k - 1, h    ));
                auto f5 = xt::eval(f(5, level, k,     h - 1));
                auto f6 = xt::eval(f(6, level, k + 1, h    ));
                auto f7 = xt::eval(f(7, level, k,     h + 1));

                auto f8  =  xt::eval(f(8, level, k - 1, h    ));
                auto f9  =  xt::eval(f(9, level, k,     h - 1));
                auto f10 = xt::eval(f(10, level, k + 1, h    ));
                auto f11 = xt::eval(f(11, level, k,     h + 1));

                auto f12 = xt::eval(f(12, level, k - 1, h    ));
                auto f13 = xt::eval(f(13, level, k,     h - 1));
                auto f14 = xt::eval(f(14, level, k + 1, h    ));
                auto f15 = xt::eval(f(15, level, k,     h + 1));

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
            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1); 
            double coeff = 1. / (1 << (2*j));

            // We take the overleaves corresponding to the existing leaves
            auto overleaves = mure::intersection(mesh[mure::MeshType::overleaves][level + 1],
                                                 mesh[mure::MeshType::cells][level]).on(level + 1);


            overleaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                // Just to provide the shape : WE CAN DO BETTER
                auto f0 = xt::eval(0.0 * f(0, level + 1, k, h));
                auto f1 = xt::eval(0.0 * f(1, level + 1, k, h));
                auto f2 = xt::eval(0.0 * f(2, level + 1, k, h));
                auto f3 = xt::eval(0.0 * f(3, level + 1, k, h));

                auto f4 = xt::eval(0.0 * f(4, level + 1, k, h));
                auto f5 = xt::eval(0.0 * f(5, level + 1, k, h));
                auto f6 = xt::eval(0.0 * f(6, level + 1, k, h));
                auto f7 = xt::eval(0.0 * f(7, level + 1, k, h));

                auto f8  = xt::eval(0.0 * f(8 , level + 1, k, h));
                auto f9  = xt::eval(0.0 * f(9 , level + 1, k, h));
                auto f10 = xt::eval(0.0 * f(10, level + 1, k, h));
                auto f11 = xt::eval(0.0 * f(11, level + 1, k, h));

                auto f12 = xt::eval(0.0 * f(12, level + 1, k, h));
                auto f13 = xt::eval(0.0 * f(13, level + 1, k, h));
                auto f14 = xt::eval(0.0 * f(14, level + 1, k, h));
                auto f15 = xt::eval(0.0 * f(15, level + 1, k, h));


                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f0  += coeff*c.second*f(0 , level + 1, k + stencil_x, h + stencil_y);
                    f4  += coeff*c.second*f(4 , level + 1, k + stencil_x, h + stencil_y);
                    f8  += coeff*c.second*f(8 , level + 1, k + stencil_x, h + stencil_y);
                    f12 += coeff*c.second*f(12, level + 1, k + stencil_x, h + stencil_y);
                    
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f1  += coeff*c.second*f(1, level + 1, k + stencil_x, h + stencil_y);
                    f5  += coeff*c.second*f(5, level + 1, k + stencil_x, h + stencil_y);
                    f9  += coeff*c.second*f(9, level + 1, k + stencil_x, h + stencil_y);
                    f13 += coeff*c.second*f(13, level + 1, k + stencil_x, h + stencil_y);

                }

                for(auto &c: pred_coeff[j][2].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f2  += coeff*c.second*f( 2, level + 1, k + stencil_x, h + stencil_y);
                    f6  += coeff*c.second*f( 6, level + 1, k + stencil_x, h + stencil_y);
                    f10 += coeff*c.second*f(10, level + 1, k + stencil_x, h + stencil_y);
                    f14 += coeff*c.second*f(14, level + 1, k + stencil_x, h + stencil_y);

                }

                for(auto &c: pred_coeff[j][3].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;
                    f3  += coeff*c.second*f( 3, level + 1, k + stencil_x, h + stencil_y);
                    f7  += coeff*c.second*f( 7, level + 1, k + stencil_x, h + stencil_y);
                    f11 += coeff*c.second*f(11, level + 1, k + stencil_x, h + stencil_y);
                    f15 += coeff*c.second*f(15, level + 1, k + stencil_x, h + stencil_y);

                }

                // // We save the fluxes
                fluxes(0, level + 1, k, h) = f0;
                fluxes(1, level + 1, k, h) = f1;
                fluxes(2, level + 1, k, h) = f2;
                fluxes(3, level + 1, k, h) = f3;

                fluxes(4, level + 1, k, h) = f4;
                fluxes(5, level + 1, k, h) = f5;
                fluxes(6, level + 1, k, h) = f6;
                fluxes(7, level + 1, k, h) = f7;

                fluxes(8,  level + 1, k, h) = f8;
                fluxes(9,  level + 1, k, h) = f9;
                fluxes(10, level + 1, k, h) = f10;
                fluxes(11, level + 1, k, h) = f11;

                fluxes(12, level + 1, k, h) = f12;
                fluxes(13, level + 1, k, h) = f13;
                fluxes(14, level + 1, k, h) = f14;
                fluxes(15, level + 1, k, h) = f15;
            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

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


                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    if (refinement(f, eps, 0.0, i))
                        break;
                }
                mure::mr_prediction_overleaves(f); // Before saving



                // if (nb_ite > 0)
                //save_solution(f, eps, nb_ite, std::string("fullcomp"));
                //save_solution(f, eps, nb_ite, std::string("nocorr"));
                //save_solution(f, eps, nb_ite);

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
