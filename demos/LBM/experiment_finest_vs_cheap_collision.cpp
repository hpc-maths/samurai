#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <xtensor/xio.hpp>

#include <mure/mr/coarsening.hpp>
#include <mure/mr/refinement.hpp>
#include <mure/mr/criteria.hpp>
#include <mure/mr/harten.hpp>
#include <mure/mr/adapt.hpp>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"

#include <chrono>



double exact_solution(double x, double t)   {

    double u = 0;

    double M = .1;

    if (x >= -1 and x < t)
    {
        u = M*(1 + x);
    }
    
    if (x >= t and x < 1)
    {
        u = M*(1 - x);
    }
       
    return u;
}

template<class Config>
auto init_f(mure::MRMesh<Config> &mesh)
{
    constexpr std::size_t nvel = 1;
    using mesh_id_t = typename mure::MRMesh<Config>::mesh_id_t;

    auto f = mure::make_field<double, nvel>("f", mesh);
    f.fill(0);

    mure::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto corner = cell.corner();
        double dx = cell.length;

        auto x = corner[0] + .5*dx;

        f[cell] = exact_solution(x, 0.0);
    });

    return f;
}

// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template<class Field, class interval_t>
xt::xtensor<double, 1> prediction_all(const Field & f, std::size_t level_g, std::size_t level,
                                      const interval_t & k,
                                      std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> & mem_map)
{

    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    auto it = mem_map.find({level_g, level, k});


    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())    {

        return it->second;
    }
    else
    {


    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    // We put only the size in x (k.size()) because in y
    // we only have slices of size 1.
    // The second term (1) should be adapted according to the
    // number of fields that we have.
    // std::vector<std::size_t> shape_x = {k.size(), 4};
    std::vector<std::size_t> shape_x = {k.size()};
    xt::xtensor<double, 1> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, k); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 1> mask_all = xt::empty<double>(shape_x);

    mask_all = mask;

    // Recursion finished
    if (xt::all(mask))
    {
        return xt::eval(f(level_g + level, k));

    }

    // If we cannot stop here

    auto kg = k >> 1;
    kg.step = 1;

    xt::xtensor<double, 1> val = xt::empty<double>(shape_x);



    auto earth  = xt::eval(prediction_all(f, level_g, level - 1, kg     , mem_map));
    auto W      = xt::eval(prediction_all(f, level_g, level - 1, kg - 1 , mem_map));
    auto E      = xt::eval(prediction_all(f, level_g, level - 1, kg + 1 , mem_map));



    // This is to deal with odd/even indices in the x direction
    std::size_t start_even = (k.start & 1) ?     1         :     0        ;
    std::size_t start_odd  = (k.start & 1) ?     0         :     1        ;
    std::size_t end_even   = (k.end & 1)   ? kg.size()     : kg.size() - 1;
    std::size_t end_odd    = (k.end & 1)   ? kg.size() - 1 : kg.size()    ;



    xt::view(val, xt::range(start_even, _, 2)) = xt::view(                        earth
                                                          + 1./8               * (W - E), xt::range(start_even, _));



    xt::view(val, xt::range(start_odd, _, 2))  = xt::view(                        earth
                                                          - 1./8               * (W - E), xt::range(_, end_odd));

    xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

    for(int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
    {
        if (mask[k_mask])
        {
            xt::view(out, k_mask) = xt::view(f(level_g + level, {k_int, k_int + 1}), 0);

        }
    }

    // It is crucial to use insert and not []
    // in order not to update the value in case of duplicated (same key)
    mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t>{level_g, level, k}
                                  ,out));


    return out;

    }
}



template<class Field, class Func>
void collision(Field &f, Func&& update_bc_for_level, bool finest_collision = false)
{

    constexpr std::size_t nvel = Field::size;

    auto mesh = f.mesh();
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto new_f = mure::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);

    if (!finest_collision)  {
        for (std::size_t level = 0; level <= max_level; ++level)    {
            auto leaves = mure::intersection(mesh[mesh_id_t::cells][level],
                                             mesh[mesh_id_t::cells][level]);
        
            leaves([&](auto &interval, auto) {
                auto k = interval;

                new_f(level, k) = xt::eval(xt::pow(xt::abs(f(level, k)), 1./5.));

            });
        }
    }

    else {

        mure::mr_projection(f);
        for (std::size_t level = mesh.min_level() - 1; level <= mesh.max_level(); ++level)
        {
            update_bc_for_level(f, level); 
        }
        mure::mr_prediction(f, update_bc_for_level);

        std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                        xt::xtensor<double, 1>> memoization_map;

        for (std::size_t level = 0; level <= max_level; ++level)    {
            
            auto leaves_on_finest = mure::intersection(mesh[mesh_id_t::cells][level],
                                                       mesh[mesh_id_t::cells][level])
                                                .on(max_level);

            leaves_on_finest([&](auto &interval, auto) {
                auto i = interval;
                auto j = max_level - level;
                
                auto f_on_finest  = prediction_all(f, level, j, i, memoization_map);

                auto f_coll = xt::eval(xt::pow(xt::abs(f_on_finest), 1./5.));

                int step = 1 << j;

                for (auto i_start = 0; i_start < (i.end - i.start); i_start = i_start + step)    {
                    
                    new_f(level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = 
                            xt::mean(xt::view(f_coll, xt::range(i_start, i_start + step)));
                }
            });
        }
    }

    std::swap(f.array(), new_f.array());
}


template<class Field, class FieldR, class Func>
double compute_error(Field &f, FieldR & fR, Func&& update_bc_for_level)
{

    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    using mesh_t = typename Field::mesh_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    update_bc_for_level(fR, max_level); // It is important to do so

    mure::mr_projection(f);
    for (std::size_t level = mesh.min_level() - 1; level <= mesh.max_level(); ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    mure::mr_prediction(f, update_bc_for_level);

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;

    
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> error_memoization_map;

    error_memoization_map.clear();

    double diff = 0.0;


    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mesh_id_t::cells][level],
                                      mesh[mesh_id_t::cells][level])
                  .on(max_level);

        exp([&](auto &interval, auto) {
            auto i = interval;
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = fR(max_level, i);

            diff  += xt::sum(xt::abs(sol - solR))[0];


        });
    }

    return dx * diff;
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
            using mesh_t = mure::MRMesh<Config>;
            using mesh_id_t = typename mesh_t::mesh_id_t;
            using coord_index_t = typename mesh_t::interval_t::coord_index_t;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = 2;//result["min_level"].as<std::size_t>();
            std::size_t max_level = 9;//result["max_level"].as<std::size_t>();

            mure::Box<double, dim> box({-3}, {3});


            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

          
                

            std::cout<<std::endl<<"Testing eps behavior"<<std::endl;
            {
                double eps = 1.;
                std::size_t N_test = 50;
                double factor = 0.80;
                std::ofstream out_eps;
                std::ofstream out_diff_ref_adap;
                std::ofstream out_compression;
                std::ofstream out_max_level;


                out_eps.open             ("./reconstructed_vs_cheap_collision/eps.dat");
                out_diff_ref_adap.open   ("./reconstructed_vs_cheap_collision/diff.dat");
                out_compression.open     ("./reconstructed_vs_cheap_collision/comp.dat");
                out_max_level.open       ("./reconstructed_vs_cheap_collision/maxlevel.dat");

                for (std::size_t n_test = 0; n_test < N_test; ++ n_test)    {
                    std::cout<<std::endl<<"Test "<<n_test<<" eps = "<<eps;

                    mesh_t mesh{box, min_level, max_level};
                    mesh_t meshR{box, max_level, max_level}; // This is the reference scheme

                    // Initialization
                    auto f  = init_f(mesh );
                    auto fR = init_f(meshR);


                    double dx = 1.0 / (1 << max_level);
                    double dt = dx; // Since lb = 1


                    auto MRadaptation = mure::make_MRAdapt(f, update_bc_for_level);
                    MRadaptation(eps, 0.);

                    for (int n = 0; n < 1; ++n) {
                        collision(f,  update_bc_for_level, true);
                        // MRadaptation(eps, 0.);

                        collision(fR, update_bc_for_level, false);
                    }


                    auto error = compute_error(f, fR, update_bc_for_level);
                        
                    std::cout<<"Diff = "<<error<<std::endl;

                    std::size_t max_level_effective = mesh.min_level();
                    for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)  {
                        if (!mesh[mesh_id_t::cells][level].empty())
                            max_level_effective = level;
                    }

                    out_max_level<<max_level_effective<<std::endl;


                    out_eps<<eps<<std::endl;
                    out_diff_ref_adap<<error<<std::endl;
                    out_compression<<static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                       / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))<<std::endl;
                    eps *= factor;
                }

                out_eps.close();
                out_diff_ref_adap.close();
                out_compression.close();
                out_max_level.close();
            }
        }
    }
    

    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
