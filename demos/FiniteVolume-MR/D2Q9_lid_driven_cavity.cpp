#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


// We use the Geier scheme because
// it seems to work pretty well


double lambda = 1.; // Lattice velocity of the scheme
double rho0 = 1.; // Reference density
double u0 = 0.05; // Reference x-velocity
double mu = 5.e-6; // Bulk viscosity
double zeta = 100. * mu; // Shear viscosity

// The relaxation parameters will be computed in the sequel
// because they depend on the space step of the scheme


template<class Config>
auto init_f(mure::Mesh<Config> &mesh)
{
    constexpr std::size_t nvel = 9;
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

        double rho = rho0;
        double qx = 0.;
        double qy = 0.;

        double cs2 = (lambda*lambda)/ 3.0; // Sound velocity of the lattice squared

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
template<class Field>
void one_time_step_overleaves_corrected(Field &f, std::size_t iter)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();
    
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
                
                // // BC
                advected(3, level, k, h) =  f(1, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);

                

            });
            
            std::cout<<std::endl<<"[=] North"<<std::flush;
            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mure::MeshType::cells);
            leaves_north.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto rho = f(0, level, k, h) + f(1, level, k, h) + f(2, level, k, h) + f(3, level, k, h) + f(4, level, k, h)
                                             + f(5, level, k, h) + f(6, level, k, h) + f(7, level, k, h) + f(8, level, k, h);


                advected(4, level, k, h) =  f(2, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h) - 0.5 * u0 * rho;
                advected(8, level, k, h) =  f(6, level, k, h) + 0.5 * u0 * rho;
                

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);




            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mure::MeshType::cells);
            leaves_northeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


                auto rho = f(0, level, k, h) + f(1, level, k, h) + f(2, level, k, h) + f(3, level, k, h) + f(4, level, k, h)
                                             + f(5, level, k, h) + f(6, level, k, h) + f(7, level, k, h) + f(8, level, k, h);


                advected(4, level, k, h) =  f(2, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h) - 0.5 * u0 * rho;
                advected(8, level, k, h) =  f(6, level, k, h) + 0.5 * u0 * rho;
                advected(3, level, k, h) =  f(3, level, k, h);
                advected(6, level, k, h) =  f(6, level, k, h);
                
                

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);


            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mure::MeshType::cells);
            leaves_west.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                advected(1, level, k, h) =  f(3, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);


            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mure::MeshType::cells);
            leaves_northwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


                auto rho = f(0, level, k, h) + f(1, level, k, h) + f(2, level, k, h) + f(3, level, k, h) + f(4, level, k, h)
                                             + f(5, level, k, h) + f(6, level, k, h) + f(7, level, k, h) + f(8, level, k, h);

                advected(1, level, k, h) =  f(3, level, k, h);
                advected(4, level, k, h) =  f(2, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);
                

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);

            });

            std::cout<<std::endl<<"[=] South"<<std::flush;
            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mure::MeshType::cells);
            leaves_south.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);


            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mure::MeshType::cells);
            leaves_southwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                

                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(1, level, k, h) =  f(3, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);

            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mure::MeshType::cells);
            leaves_southeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(3, level, k, h) =  f(1, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);
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

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);
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
                auto f8 = xt::eval(advected(8, level, k, h));
              
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

                double space_step = 1.0 / (1 << max_level);

                double dummy = 3.0/(lambda*rho0*space_step);
                double sigma_1 = dummy*(zeta - 2.*mu/3.);
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
    str << "LBM_D2Q9_Lid_Driven_Cavity_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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


    h5file.add_field(rho);
    h5file.add_field(qx);
    h5file.add_field(qy);
    h5file.add_field(vel_mod);

    h5file.add_field(f);
    h5file.add_field(level_);

}

int main(int argc, char *argv[])
{
    cxxopts::Options options("D2Q9 scheme for the simulation of the Von Karman vortex street",
                             "We will add multiresolution very soon");

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
            mure::Mesh<Config> mesh{box, max_level, max_level};


            auto f = init_f(mesh);

            double T = 1000.;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout<<std::endl<<"Iteration number = "<<nb_ite<<std::endl;

                if (nb_ite % 32)
                    save_solution(f, eps, nb_ite/32, std::string("_before")); // Before applying the scheme

                one_time_step_overleaves_corrected(f, nb_ite);

            }
    
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
