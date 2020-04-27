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
    u = exp(-20.0 * (x) * (x));

    return u;
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


template<class Field, class interval_t>
xt::xtensor<double, 2> prediction_all(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, 
                                  std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 2>> & mem_map)
{

    using namespace xt::placeholders;
    // We check if the element is already in the map
    auto it = mem_map.find({level_g, level, i});
    if (it != mem_map.end())
    {
        return it->second;
    }
    else
    {
        auto mesh = f.mesh();
        std::vector<std::size_t> shape = {i.size(), 1};
        xt::xtensor<double, 2> out = xt::empty<double>(shape);
        auto mask = mesh.exists(level_g + level, i, mure::MeshType::cells_and_ghosts);

        xt::xtensor<double, 2> mask_all = xt::empty<double>(shape);
        xt::view(mask_all, xt::all(), 0) = mask;

        if (xt::all(mask))
        {         
            return xt::eval(f(level_g + level, i));
        }

        auto ig = i >> 1;
        ig.step = 1;

        xt::xtensor<double, 2> val = xt::empty<double>(shape);
        auto current = xt::eval(prediction_all(f, level_g, level-1, ig, mem_map));
        auto left = xt::eval(prediction_all(f, level_g, level-1, ig-1, mem_map));
        auto right = xt::eval(prediction_all(f, level_g, level-1, ig+1, mem_map));

        std::size_t start_even = (i.start&1)? 1: 0;
        std::size_t start_odd = (i.start&1)? 0: 1;
        std::size_t end_even = (i.end&1)? ig.size(): ig.size()-1;
        std::size_t end_odd = (i.end&1)? ig.size()-1: ig.size();
        xt::view(val, xt::range(start_even, _, 2)) = xt::view(current - 1./8 * (right - left), xt::range(start_even, _));
        xt::view(val, xt::range(start_odd, _, 2)) = xt::view(current + 1./8 * (right - left), xt::range(_, end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, ++i_int)
        {
            if (mask[i_mask])
            {
                xt::view(out, i_mask) = xt::view(f(level_g + level, {i_int, i_int + 1}), 0);
            }
        }

        // The value should be added to the memoization map before returning
        return out;// mem_map[{level_g, level, i, ig}] = out;
    }
}


template<class Config, class FieldR>
double compute_error(mure::Field<Config, double, 1> &f, FieldR & fR, double t)
{

    auto mesh = f.mesh();
    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    mure::mr_projection(f);
    mure::mr_prediction(f);  // C'est supercrucial de le faire.

    f.update_bc(); // Important especially when we enforce Neumann...for the Riemann problem
    fR.update_bc();    

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;
    error_memoization_map.clear();

    double error = 0; // To return

    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(meshR[mure::MeshType::cells][max_level],
                                      mesh[mure::MeshType::cells][level])
                  .on(max_level);

        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));


            error += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), xt::range(0, 1))) - xt::flatten(xt::view(fR(max_level, i), xt::all(), xt::range(0, 1)))))[0];


        });
    }

    return dx * error; // Normalization by dx before returning
    // I think it is better to do the normalization at the very end ... especially for round-offs    
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
            mure::Mesh<Config> meshR{box, max_level, max_level};

            // Initialization
            auto f   = init_f(mesh , 0.0);
            auto fR  = init_f(meshR , 0.0);


            double neglected_terms = 0.0;

            for (std::size_t ite=0; ite<max_level-min_level; ++ite) {

                //  using Config = typename Field::Config;
                // using value_type = typename Field::value_type;
                // constexpr auto size = Field::size;
                // constexpr auto dim = Config::dim;
                // constexpr auto max_refinement_level = Config::max_refinement_level;
                // using interval_t = typename Config::interval_t;

                auto mesh = f.mesh();
                std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

                mure::Field<Config, double, 1> detail{"detail", mesh};

                mure::Field<Config, int, 1> tag{"tag", mesh};
                tag.array().fill(0);
                mesh.for_each_cell([&](auto &cell) {
                    tag[cell] = static_cast<int>(mure::CellFlag::keep);
                });

                mure::mr_projection(f);
                mure::mr_prediction(f);
                f.update_bc();


                // Cela ne nous sert a rien ... Mais on le garde pour ne rien changer apres
                typename std::conditional<1,
                                          xt::xtensor_fixed<double, xt::xshape<16 + 1>>,
                                          xt::xtensor_fixed<double, xt::xshape<16 + 1, 1>>
                                         >::type max_detail;
                max_detail.fill(std::numeric_limits<double>::min());


                // What are the data it uses at min_level - 1 ???
                for (std::size_t level = min_level - 1; level < max_level - ite; ++level)   {
                    auto subset = intersection(mesh[mure::MeshType::all_cells][level],
                                               mesh[mure::MeshType::cells][level + 1])
                                 .on(level);
                    subset.apply_op(level, compute_detail(detail, f), compute_max_detail(detail, max_detail));
                }



                // AGAIN I DONT KNOW WHAT min_level - 1 is
                for (std::size_t level = min_level; level <= max_level - ite; ++level)
                {
                    int exponent = dim * (level - max_level);

                    auto eps_l = std::pow(2, exponent) * eps;

                    // COMPRESSION

                    auto subset_1 = mure::intersection(mesh[mure::MeshType::cells][level],
                                                       mesh[mure::MeshType::all_cells][level-1])
                                   .on(level-1);


                    // This operations flags the cells to coarsen
                    subset_1.apply_op(level, to_coarsen_mr(detail, max_detail, tag, eps_l, min_level));
                    //subset_1.apply_op(level, to_coarsen_mr_BH(detail, max_detail, tag, eps_l, min_level));

                    auto subset_2 = intersection(mesh[mure::MeshType::cells][level],
                                                 mesh[mure::MeshType::cells][level]);
                    auto subset_3 = intersection(mesh[mure::MeshType::cells_and_ghosts][level],
                                                 mesh[mure::MeshType::cells_and_ghosts][level]);

                    subset_2.apply_op(level, mure::enlarge(tag, mure::CellFlag::keep));
                    subset_3.apply_op(level, mure::tag_to_keep(tag));


                    // I now sum the discarded details with a weight

                    double weight = pow(2.0, - static_cast<double>(level));

                    subset_1([&](auto, auto &interval, auto) {
                        auto i = interval[0];

                        auto mask = (tag(level, i) <= static_cast<int>(mure::CellFlag::coarsen)) and
                                    (tag(level, i) >= static_cast<int>(mure::CellFlag::coarsen));

                        xt::masked_view(detail(level, i), !mask) = 0.0;

                        auto tmp = xt::sum(xt::abs(detail(level, i)));

                        //std::cout<<std::endl<<"Level = "<<level<<" Neg detaims "<<xt::masked_view(detail(level, i), mask)<<std::endl<<" Mask "<<mask;

                        //std::cout<<std::endl<<"Level = "<<level<<" What "<<tmp;
                                              //<<std::endl<<"Masked val = "<<cmp;

                        //auto help = xt::view(neglected_details, xt::all(), xt::range(0, 1));

                        neglected_terms += weight * tmp[0];

                    });
                }


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



            std::cout<<std::endl<<"Norm neglected terms (/2) = "<<neglected_terms / 2.0;


            auto error = compute_error(f, fR, 0.0);

        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }


    std::cout<<std::endl;



    return 0;
}
