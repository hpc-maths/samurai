#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
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

        double m0 = 0;

        double radius = .1;
        double x_center = 0.5, y_center = 0.75;
        if ((   (x - x_center) * (x - x_center) + 
                (y - y_center) * (y - y_center))
                <= radius * radius)
            m0 = 1;
        // if (abs(x - x_center) <= radius and abs(y - y_center) <= radius)    {
        //     m0 = 1.;
        // }


        double m1 = 0.5 * kx * m0 * m0;
        double m2 = 0.5 * ky * m0 * m0;
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

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(8);
        for (int l = 0; l < size; ++l)
        {
            // Be careful, there is no sign on this fluxes

            // Along x (vertical edge)
            data[k][0] += prediction(k, i*size - 1, j*size + l); // In W
            data[k][1] += prediction(k, (i+1)*size - 1, j*size + l); // Out E
            data[k][2] += prediction(k, (i+1)*size, j*size + l); // In E
            data[k][3] += prediction(k, i*size, j*size + l); // Out W

            // Along y (horizontal edge)
            data[k][4] += prediction(k, i*size + l, j*size - 1); // In S
            data[k][5] += prediction(k, i*size + l, (j+1)*size - 1); // Out N
            data[k][6] += prediction(k, i*size + l, (j+1)*size); // In N
            data[k][7] += prediction(k, i*size + l, j*size); // Our S
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
    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * yp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * yp))), // Removing SE
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_north(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * xp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * xp))), // Removing NW
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_west(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * yp))), // Removing NW
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * yp))), // Removing SW
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_south(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * xp))), // Removing SE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * xp))), // Removing SW
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * d11)),
                                              translate(mesh.initial_mesh(), - 1 * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), - 1 * xp)), // Removing horizontal strip
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * d1m1)),
                                              translate(mesh.initial_mesh(), - 1 * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), 1 * xp)), // Removing horizontal strip
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), 1 * d11)),
                                              translate(mesh.initial_mesh(), 1 * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), 1 * xp)), // Removing horizontal strip
                        mesh[type][level]).on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -1 * d1m1)),
                                              translate(mesh.initial_mesh(), 1 * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), -1 * xp)), // Removing horizontal strip
                        mesh[type][level]).on(level);
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


    Field advected{"advected", mesh};
    advected.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        if (level == max_level) {


            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, mure::MeshType::cells);
            leaves_east([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    // advected(2 + 4 * scheme_n, level, k, h) =  -1.*f(0 + 4 * scheme_n, level, k    , h    ); // Direct evaluation of the BC
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); // Direct evaluation of the BC
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });
                
            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mure::MeshType::cells);
            leaves_north([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1 ,h    ); 
                    // advected(3 + 4 * scheme_n, level, k, h) =  -1.*f(1 + 4 * scheme_n, level, k,     h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );

                }
            });

            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mure::MeshType::cells);
            leaves_northeast([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    // advected(2 + 4 * scheme_n, level, k, h) =  -1.*f(0 + 4 * scheme_n, level, k    , h    ); 
                    // advected(3 + 4 * scheme_n, level, k, h) =  -1.*f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );
                }
            });

            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mure::MeshType::cells);
            leaves_west([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    // advected(0 + 4 * scheme_n, level, k, h) =  -1.*f(2 + 4 * scheme_n, level, k    , h    );
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mure::MeshType::cells);
            leaves_northwest([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    // advected(0 + 4 * scheme_n, level, k, h) =  -1.*f(2 + 4 * scheme_n, level, k    , h    );
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    // advected(3 + 4 * scheme_n, level, k, h) =  -1.*f(1 + 4 * scheme_n, level, k,     h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h    );

                }
            });

            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mure::MeshType::cells);
            leaves_south([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    // advected(1 + 4 * scheme_n, level, k, h) =  -1.*f(3 + 4 * scheme_n, level, k,     h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mure::MeshType::cells);
            leaves_southwest([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    // advected(0 + 4 * scheme_n, level, k, h) =  -1.*f(2 + 4 * scheme_n, level, k    , h    );
                    // advected(1 + 4 * scheme_n, level, k, h) =  -1.*f(3 + 4 * scheme_n, level, k,     h    );
                    advected(0 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mure::MeshType::cells);
            leaves_southeast([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 1; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    // advected(1 + 4 * scheme_n, level, k, h) =  -1.*f(3 + 4 * scheme_n, level, k,     h    );
                    // advected(2 + 4 * scheme_n, level, k, h) =  -1.*f(0 + 4 * scheme_n, level, k    , h    ); 
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k    , h    ); 
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            // Advection far from the boundary
            auto tmp1 = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2 = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);

            auto internal_leaves = difference(mesh[mure::MeshType::cells][max_level],
                                      all_leaves_boundary);
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


                m0_1 = (1. - sq) *  kx + sq * 0.5 * m0_0 * m0_0;
                m0_2 = (1. - sq) *  ky + sq * 0.5 * m0_0 * m0_0;
                m0_3 = (1. - sxy) * m0_3; 



                new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;


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
