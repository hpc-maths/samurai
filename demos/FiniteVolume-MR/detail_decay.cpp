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

double function_to_compress(double x)   {
    double u = 0;


    double t = 1.0;
    //return (x <= x0 + vshock * t) ? rhoL : rhoR;
    
    // if (x >= -1 and x < t)
    // {
    //     u = (1 + x) / (1 + t);
    // }
    
    // if (x >= t and x < 1)
    // {
    //     u = (1 - x) / (1 - t);
    // }

    if (x > -1.0) {
        if (x < 0.) {
            u = 0.5 + x*(1+x/2);
        }
        else {
            if (x < 1){
                u = 0.5 + x*(1-x/2);
            }
            else
            {
                u = 1.;
            }
            
        }
    }

    //u = exp(-20.0 * (x) * (x));
    return u;
}

double flux(double u)   {
    return 0.5 * u * u;
}

template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 1;
    mure::BC<1> bc{ {{ {mure::BCType::dirichlet, 0},
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        f[cell][0] = function_to_compress(x) ;
    });

    return f;
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

int main(int argc, char *argv[])
{
    cxxopts::Options options("study of the detail decay...",
                             "...");

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
            using Config = mure::MRConfig<dim, 1>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();
            double s = result["s"].as<double>();


            mure::Box<double, dim> box({-3}, {3});
            mure::Mesh<Config> mesh{box, min_level, max_level};
            mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

            // Initialization
            auto f  = init_f(mesh , 0.0);
            
            
            double regularity_index = 3.0; // What is the estimated Sobolev regularity of the function
            double old_max_det = 0.0;
            
            for (std::size_t i=0; i < max_level - min_level; ++i)
            {

                //using Config = typename mure::Field::Config;
                //mure::Field<Config, double, 1> detail{"detail", mesh};
                auto detail = f; 


                mure::mr_projection(f);
                mure::mr_prediction(f);
                f.update_bc();



                for (std::size_t level = min_level - 1; level < max_level - i; ++level)   {

                    auto subset = intersection(mesh[mure::MeshType::all_cells][level],
                                               mesh[mure::MeshType::cells][level + 1]).on(level);
                    
                    
                    subset.apply_op(level, compute_detail(detail, f));


                    subset([&](auto, auto &interval, auto) {
                        auto i = interval[0];

                        auto det_on_l_p_1 = xt::abs(xt::eval(detail(level + 1, i)));

                        //auto max_det = xt::maximum(xt::abs(det_on_l_p_1[0]));

                        double maxdet = 0.0;

                        for (auto el : det_on_l_p_1)    {
                            if (el > maxdet)
                                maxdet = el;
                        }
                        
                        std::cout<<std::endl<<"Level = "<<level + 1<<"   Max detail = "<<maxdet<<" Actual ratio = "<<maxdet / old_max_det;

                        old_max_det = maxdet;

                    });

                }
                


                // We eliminate all the level since we have has what we were looking for ...


                mure::Field<Config, int, 1> tag{"tag", mesh};
                tag.array().fill(0);
                mesh.for_each_cell([&](auto &cell) {
                    tag[cell] = static_cast<int>(mure::CellFlag::coarsen);
                });

                // FROM NOW ON LOIC HAS TO EXPLAIN
                for (std::size_t level = max_level; level > 0; --level)
                {
                    auto keep_subset = intersection(mesh[mure::MeshType::cells][level],
                                                    mesh[mure::MeshType::all_cells][level - 1])
                                      .on(level - 1);
                    keep_subset.apply_op(level - 1, maximum(tag));

                    xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        stencil.fill(0);
                        for (int s = -1; s <= 1; ++s)
                        {
                            if (s != 0)
                            {
                                stencil[d] = s;
                                auto subset = intersection(mesh[mure::MeshType::cells][level],
                                                           translate(mesh[mure::MeshType::cells][level - 1], stencil))
                                             .on(level - 1);
                                subset.apply_op(level - 1, balance_2to1(tag, stencil));
                            }
                        }
                    }
                }

                mure::CellList<Config> cell_list;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto level_cell_array = mesh[mure::MeshType::cells][level];
                    if (!level_cell_array.empty())
                    {
                        level_cell_array.for_each_interval_in_x([&](auto const &index_yz, auto const &interval) {
                            for (int i = interval.start; i < interval.end; ++i)
                            {
                                if (tag.array()[i + interval.index] & static_cast<int>(mure::CellFlag::keep))
                                    {
                                        cell_list[level][index_yz].add_point(i);
                                    }
                                    else
                                    {
                                        cell_list[level-1][index_yz>>1].add_point(i>>1);
                                    }
                                }
                            });
                    }
                }

                mure::Mesh<Config> new_mesh{cell_list, mesh.initial_mesh(),
                                        min_level, max_level};

    

                mure::Field<Config, double, 1> new_f{f.name(), new_mesh, f.bc()};

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto subset = mure::intersection(mesh[mure::MeshType::all_cells][level],
                                               new_mesh[mure::MeshType::cells][level]);
                    subset.apply_op(level, copy(new_f, f));
                }

                f.mesh_ptr()->swap(new_mesh);
                std::swap(f.array(), new_f.array());

            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }


    std::cout<<std::endl;



    return 0;
}
