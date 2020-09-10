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
double prediction_all(const Field & f, std::size_t level_g, std::size_t level, 
                      const interval_t & i, const ordinates_t & j)
{

    auto mesh = f.mesh();


    std::vector<std::size_t> shape_x = {i.size(), 2};

    std::cout<<std::endl<<"Level_g = "<<level_g<<"  level = "<<level<<"  i_size = "<<i.size()<<"  i = "<<i<<"  j = "<<j<<std::flush;


    return 1.;



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
            auto h = interval[0];
            auto k = index[0];

            auto tmp = prediction_all(f, level, max_level - level, h, k);


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
