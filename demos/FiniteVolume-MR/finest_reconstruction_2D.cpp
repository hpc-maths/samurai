#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{

    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0}
                    }} };

    mure::Field<Config, double, 1> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double f_new = (y < 0.5) ? 0.5 : 1.;

        if (std::sqrt(std::pow(x - .5, 2.) + std::pow(y - .5, 2.)) < 0.15)  {
            f_new = 2.;
        }

        f[cell][0] = f_new;


    });

    return f;
}


template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "Finest_Reconstruction_2D_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
    });

    h5file.add_field(f);
    h5file.add_field(level_);
}


template<class Field, class interval_t, class ordinates_t>
xt::xtensor<double, 1> prediction_all(const Field & f, std::size_t level_g, std::size_t level, 
                                      const interval_t & k, const ordinates_t & h)
{

    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    std::cout<<std::endl<<"Before doing - k = "<<k<<std::flush;

    auto mesh = f.mesh();

    // We put only the size in x (k.size()) because in y
    // we only have slices of size 1. 
    // The second term (1) should be adapted according to the 
    // number of fields that we have.
    std::vector<std::size_t> shape_x = {k.size(), 1};
    xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(mure::MeshType::cells_and_ghosts, level_g + level, k, h); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);
        
    xt::view(mask_all, xt::all(), 0) = mask; // We have only this because we only have one field

    // Recursion finished
    if (xt::all(mask))
    {         
        return xt::eval(f(level_g + level, k, h));
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

   std::cout<<std::endl<<"In pred_all - level_g = "<<level_g<<"  level = "<<level<<"  k = "<<k<<"  h = "<<h<<"   kg = "<<kg<<std::flush;

    auto earth  = xt::eval(prediction_all(f, level_g, level - 1, kg    , h / 2    ));
    auto W      = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, h / 2    ));
    auto E      = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, h / 2    ));
    auto S      = xt::eval(prediction_all(f, level_g, level - 1, kg    , h / 2 - 1));
    auto N      = xt::eval(prediction_all(f, level_g, level - 1, kg    , h / 2 + 1));
    auto SW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, h / 2 - 1));
    auto SE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, h / 2 - 1));
    auto NW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, h / 2 + 1));
    auto NE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, h / 2 + 1));

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

    // // Gives a segfault as well

    // xt::view(val, xt::range(start_even, _, 2)) = xt::view(earth - 1./8 * (E - W), xt::range(start_even, _));
    // xt::view(val, xt::range(start_odd, _, 2))  = xt::view(earth + 1./8 * (E - W), xt::range(_, end_odd));


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
            xt::view(out, k_mask) = xt::view(f(level_g + level, {k_int, k_int + 1}, h), 0);
        }
    }

    return out;
}

template<class Field>
void foo(Field & f)
{

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves_on_finest = mure::intersection(mesh[mure::MeshType::cells][level],
                                                   mesh[mure::MeshType::cells][level]);
        
        leaves_on_finest.on(max_level)([&](auto& index, auto &interval, auto) {
            auto k = interval[0];
            auto h = index[0];

            std::cout<<std::endl<<"In foo - level = "<<level<<"  k = "<<k<<"  h = "<<h<<std::flush;

            auto tmp = prediction_all(f, level, max_level - level, k, h);


        });

    }
}



int main(int argc, char *argv[])
{
    cxxopts::Options options("...",
                             "...");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("10"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
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

            auto f = init_f(mesh, 0);

            auto mesh_everywhere_refined(mesh);
            auto f_everywhere_refined = init_f(mesh_everywhere_refined, 0);

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

            mure::mr_prediction_overleaves(f);

            save_solution(f, eps, 0);
            save_solution(f_everywhere_refined, 0., 0);


            foo(f);

            
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
