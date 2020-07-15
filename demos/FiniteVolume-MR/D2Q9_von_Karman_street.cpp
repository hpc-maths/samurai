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


double radius = 1./32.; // Radius of the obstacle
double lambda = 1.; // Lattice velocity of the scheme
double rho0 = 1.; // Reference density
double u0 = 0.05; // Reference x-velocity
double mu = 5.e-6; // Bulk viscosity
double zeta = 10. * mu; // Shear viscosity

// The relaxation parameters will be computed in the sequel
// because they depend on the space step of the scheme


// This construct the mesh with a hole inside corresponding 
// to the obstacle
template<class Config>
auto build_mesh(std::size_t min_level, std::size_t max_level)
{
    constexpr std::size_t dim = Config::dim;

    mure::Box<double, dim> box({0, 0}, {2, 1});
    mure::Mesh<Config> mesh{box, min_level, max_level};

    mure::CellList<Config> cl;
    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double radius = 1./32.;
        double x_center = 5./16., y_center = 0.5;

        if ((   std::max(std::abs(x - x_center), 
                std::abs(y - y_center)))
                > radius)
        {
            cl[cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
        }
    });

    // It is important to add the initial mesh argument
    // in order to have it available during the simuation, especially
    // to deal with the boundary
    mure::Mesh<Config> new_mesh(cl, mesh.initial_mesh(), min_level, max_level);
    return new_mesh;
}


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
        double qx = rho0 * u0;
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
                        mesh[type][level]);        
}

template<class Mesh> 
auto get_adjacent_hole_west(Mesh & mesh)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -xp)),
                                              difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -yp))),
                                   difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], yp))),
                        contraction(mesh.initial_mesh()));        
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

// template<class Mesh> 
// auto get_adjacent_hole_south(Mesh & mesh)
// {
//     const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
//     const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

//     return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -yp)),
//                                               difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -xp))),
//                                    difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], xp))),
//                         contraction(mesh.initial_mesh()));        
// }

template<class Mesh> 
auto get_adjacent_hole_south(Mesh & mesh)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -yp)),
                                              difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -xp))),
                                   difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], xp))),
                        contraction(mesh.initial_mesh()));        
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
auto get_adjacent_hole_east(Mesh & mesh)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], xp)),
                                              difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -yp))),
                                   difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], yp))),
                        contraction(mesh.initial_mesh()));        
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
auto get_adjacent_hole_north(Mesh & mesh)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], yp)),
                                              difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], -xp))),
                                   difference(mesh.initial_mesh(), translate(mesh[mure::MeshType::cells][mesh.max_level()], xp))),
                        contraction(mesh.initial_mesh()));        
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


template<class Field>
void verification_boundary(Field &f)
{

    std::cout<<std::endl<<"Verification ==== "<<std::endl;

    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();






    auto full = intersection(mesh.initial_mesh(), mesh.initial_mesh());
    full.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"Full k = "<<k<<" h = "<<h<<std::endl;
            
    }); 


    std::cout<<std::endl<<"Reduced"<<std::endl<<std::endl;
    auto red = intersection(mesh.initial_mesh(), contraction(mesh.initial_mesh()));
    red.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"Red k = "<<k<<" h = "<<h<<std::endl;
            
    });

    return;



    auto leaves_west_hole = get_adjacent_hole_west(mesh);
    leaves_west_hole.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"West hole last cells k = "<<k<<" h = "<<h<<std::endl;
            
    }); 

    auto leaves_east_hole = get_adjacent_hole_east(mesh);
    leaves_east_hole.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"East hole last cells k = "<<k<<" h = "<<h<<std::endl;
            
    }); 

    auto leaves_east_north = get_adjacent_hole_north(mesh);
    leaves_east_north.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"North hole last cells k = "<<k<<" h = "<<h<<std::endl;
            
    }); 

    auto leaves_east_south = get_adjacent_hole_south(mesh);
    leaves_east_south.on(max_level)([&](auto& index, auto &interval, auto) {
        auto k = interval[0]; // Logical index in x
        auto h = index[0];    // Logical index in y 
        
        std::cout<<std::endl<<"South hole last cells k = "<<k<<" h = "<<h<<std::endl;
            
    }); 
}


template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q9_von_Karman_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
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



            auto mesh = build_mesh<Config>(min_level, max_level);

            auto f = init_f(mesh);

            verification_boundary(f);

            save_solution(f, eps, 0);

        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
