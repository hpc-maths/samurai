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


#include <chrono>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"




template<class coord_index_t>
auto compute_prediction_separate_inout(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(4);

        data[k][0] = prediction(k, i*size - 1);
        data[k][1] = prediction(k, (i+1)*size - 1);
        data[k][2] = prediction(k, (i+1)*size);
        data[k][3] = prediction(k, i*size);
    }
    return data;
}


double exact_solution(double x, double t)   {

    double u = 0;

    if (x >= -1 and x < t)
    {
        u = (1 + x) / (1 + t);
    }
    
    if (x >= t and x < 1)
    {
        u = (1 - x) / (1 - t);
    }

    return u;
    // return std::exp(-30.*x*x);
}


template<class Config>
auto init_f(mure::MRMesh<Config> & mesh)
{
    
    constexpr std::size_t nvel = 2;
    using mesh_id_t = typename mure::MRMesh<Config>::mesh_id_t;

    auto f = mure::make_field<double, nvel>("f", mesh);
    f.fill(0);

    mure::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {

        auto corner = cell.corner();
        double dx = cell.length;

        auto x = corner[0] + .5 * dx;
        double u = 0;

        u = exact_solution(x, 0.0);
        double v = .5*u*u;

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
    });

    return f;
}


// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template<class Field, class interval_t>
xt::xtensor<double, 2> prediction_all(const Field & f, std::size_t level_g, std::size_t level,
                                      const interval_t & k,
                                      std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> & mem_map)
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
    std::vector<std::size_t> shape_x = {k.size(), 2};
    xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, k); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);

    // for (int h_field = 0; h_field < 4; ++h_field)  {
    for (int h_field = 0; h_field < 2; ++h_field)  {
        xt::view(mask_all, xt::all(), h_field) = mask;
    }

    // Recursion finished
    if (xt::all(mask))
    {
        return xt::eval(f(0, 2, level_g + level, k));

    }

    // If we cannot stop here

    auto kg = k >> 1;
    kg.step = 1;

    xt::xtensor<double, 2> val = xt::empty<double>(shape_x);



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
            xt::view(out, k_mask) = xt::view(f(0, 2, level_g + level, {k_int, k_int + 1}), 0);

        }
    }

    // It is crucial to use insert and not []
    // in order not to update the value in case of duplicated (same key)
    mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t>{level_g, level, k}
                                  ,out));


    return out;

    }
}


template<class Field, class Func, class Pred>
void one_time_step_overleaves(Field &f, Func&& update_bc_for_level, 
                            const Pred& pred_coeff, double s_rel, 
                            int finest_collision = 0)
{


    double lambda = 1.;

    constexpr std::size_t nvel = Field::size;

    auto mesh = f.mesh();
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    mure::mr_projection(f);
    for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    mure::mr_prediction(f, update_bc_for_level);


    // After that everything is ready, we predict what is remaining
    mure::mr_prediction_overleaves(f, update_bc_for_level);

    auto new_f = mure::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);

    auto advected_f = mure::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);

    auto help_f = mure::make_field<double, nvel>("help_f", mesh);
    help_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        // If we are at the finest level, we no not need to correct

        if (level == max_level) {
            std::size_t j = 0;
            double coeff = 1.;


            auto leaves = mure::intersection(mesh[mesh_id_t::cells][max_level],
                                             mesh[mesh_id_t::cells][max_level]);
            leaves([&](auto &interval, auto) {

                auto k = interval;

                advected_f(0, max_level, k) = xt::eval(f(0, max_level, k - 1));
                advected_f(1, max_level, k) = xt::eval(f(1, max_level, k + 1));

            });
        }

        // Otherwise, correction is needed
        else
        {

            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1);
            double coeff = 1. / (1 << j);


            // We take the overleaves corresponding to the existing leaves
            // auto overleaves = mure::intersection(mesh[mesh_id_t::cells][level],
            //                                      mesh[mesh_id_t::cells][level]).on(level + 1);

            auto ol = mure::intersection(mesh[mesh_id_t::cells][level],
                                         mesh[mesh_id_t::cells][level]).on(level + 1);

            ol([&](auto& interval, auto) {
                auto k = interval; // Logical index in x



                auto fp = xt::eval(f(0, level + 1, k));
                auto fm = xt::eval(f(1, level + 1, k));



                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += coeff * weight * f(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp -= coeff * weight * f(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][2].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fm += coeff * weight * f(1, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][3].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fm -= coeff * weight * f(1, level + 1, k + stencil);
                }

                // Save it
                help_f(0, level + 1, k) = fp;
                help_f(1, level + 1, k) = fm;

            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = mure::intersection(mesh[mesh_id_t::cells][level],
                                             mesh[mesh_id_t::cells][level]);

            leaves([&](auto &interval, auto) {
                auto k = interval;

                // Projection
                advected_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1)));
                advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1)));

            });
        }
    }


    // Collision


    if (finest_collision == 0)  {


        for (std::size_t level = 0; level <= max_level; ++level)    {

            double dx = 1./(1 << level);

            auto leaves = mure::intersection(mesh[mesh_id_t::cells][level],
                                                 mesh[mesh_id_t::cells][level]);
        

            leaves([&](auto &interval, auto) {
                auto k = interval;
                auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
                auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

                vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(uu, 2.);

                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);

            });
        }
    }

    if (finest_collision == 1)  {

        mure::mr_projection(advected_f);
        for (std::size_t level = min_level - 1; level <= max_level; ++level)
        {
            update_bc_for_level(advected_f, level); // It is important to do so
        }
        mure::mr_prediction(advected_f, update_bc_for_level);

        for (std::size_t level = 0; level <= max_level; ++level)    {

            double dx = 1./(1 << level);

            auto leaves = mure::intersection(mesh[mesh_id_t::cells][level],
                                                 mesh[mesh_id_t::cells][level]);
        

            leaves([&](auto &interval, auto) {
                auto k = interval;
                auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
                auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

                if (level < max_level)  {
                    // We compute the cells centers
                    auto uum1 = xt::eval(advected_f(0, level, k-1) + advected_f(1, level, k-1));
                    auto uup1 = xt::eval(advected_f(0, level, k+1) + advected_f(1, level, k+1));


                    auto cc = dx*(k.start + 0.5) + dx * xt::arange(k.size());
                    auto cl = dx*(k.start) + dx * xt::arange(k.size());
                    auto cr = dx*(k.start + 1.) + dx * xt::arange(k.size());


                    auto pol = [&](auto point)
                    {
                        return (.5/(dx*dx)*(uum1+uup1)-1./(dx*dx)*uu) * xt::pow(point, 2.)
                                + (-(.5*dx+cc)/(dx*dx)*uum1+2.*cc/(dx*dx)*uu+(.5*dx-cc)/(dx*dx)*uup1) * point
                                + ((-1./24+.5*xt::pow(cc, 2.)/(dx*dx))*(uum1+uup1)+(.5*cc/dx)*(uum1-uup1)+(13./12-xt::pow(cc, 2.)/(dx*dx))*uu);
                    };

                    auto pol_primitive = [&](auto point)
                    {
                        return 1./3*(.5/(dx*dx)*(uum1+uup1)-1./(dx*dx)*uu) * xt::pow(point, 3.)
                                +1./2*(-(.5*dx+cc)/(dx*dx)*uum1+2.*cc/(dx*dx)*uu+(.5*dx-cc)/(dx*dx)*uup1) * xt::pow(point, 2.)
                                + ((-1./24+.5*xt::pow(cc, 2.)/(dx*dx))*(uum1+uup1)+(.5*cc/dx)*(uum1-uup1)+(13./12-xt::pow(cc, 2.)/(dx*dx))*uu)*point;
                    };


                    auto sliding_average = [&] (auto p_left, auto p_right, double dx_loc)
                    {
                        return 1./dx_loc * (pol_primitive(p_right) - pol_primitive(p_left));
                    };


                    // vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(pol(cc), 2.); // Midpoint
                    vv = (1 - s_rel) * vv + s_rel * .5 * (.5*xt::pow(pol(cl), 2.) + .5*xt::pow(pol(cr), 2.)); // Trapezoidal


                }
                else
                {
                    vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(uu, 2.);

                }

                // vv = (1. - s_rel)*vv + s_rel * .5 * uu * uu;

                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);

            });
        }
    }


    if (finest_collision == 2) {

        mure::mr_projection(advected_f);
        for (std::size_t level = mesh.min_level() - 1; level <= mesh.max_level(); ++level)
        {
            update_bc_for_level(advected_f, level); 
        }
        mure::mr_prediction(advected_f, update_bc_for_level);


            
        std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                        xt::xtensor<double, 2>> memoization_map;

        for (std::size_t level = 0; level <= max_level; ++level)    {
                
                
                
            // std::cout<<std::endl<<"Level = "<<level<<std::endl;


            auto leaves_on_finest = mure::intersection(mesh[mesh_id_t::cells][level],
                                                       mesh[mesh_id_t::cells][level])
                                                .on(max_level);

            leaves_on_finest([&](auto &interval, auto) {
                auto i = interval;
                auto j = max_level - level;
                
                auto f_on_finest  = prediction_all(advected_f, level, j, i, memoization_map);


                auto uu = xt::eval(xt::view(f_on_finest, xt::all(), 0) 
                                 + xt::view(f_on_finest, xt::all(), 1));

                auto vv = xt::eval(lambda*(xt::view(f_on_finest, xt::all(), 0) 
                                         - xt::view(f_on_finest, xt::all(), 1))); 

                vv = (1. - s_rel)*vv + s_rel * .5 * uu * uu;

                auto f_0_post_coll = .5 * (uu + 1. / lambda * vv);
                auto f_1_post_coll = .5 * (uu - 1. / lambda * vv);

                // std::cout<<std::endl<<"i = "<<i<<" Tableau = "<<f_0_post_coll<<std::endl;



                int step = 1 << j;

                for (auto i_start = 0; i_start < (i.end - i.start); i_start = i_start + step)    {

                    // std::cout<<"Mean = "<<xt::mean(xt::view(f_0_post_coll, xt::range(i_start, i_start + step)))<<std::endl;
                    
                    new_f(0, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_0_post_coll, xt::range(i_start, i_start + step)));
                    new_f(1, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_1_post_coll, xt::range(i_start, i_start + step)));
                }
            });
        }
    }

    std::swap(f.array(), new_f.array());
}



template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext)
{
    using value_t = typename Field::value_type;
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D1Q2_Burgers_finest_reconstruction_collision_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto level = mure::make_field<std::size_t, 1>("level", mesh);
    auto u = mure::make_field<value_t, 1>("u", mesh);

    mure::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        level[cell] = static_cast<double>(cell.level);
        u[cell] = f[cell][0] + f[cell][1];
    });

    mure::save(str.str().data(), mesh, u, f, level);

}





int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers",
                             "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("9"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
                       ("s", "relaxation parameter", cxxopts::value<double>()->default_value("0.75"))
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
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();
            double s = result["s"].as<double>();


            mure::Box<double, dim> box({-3}, {3});
            mure::MRMesh<Config> mesh {box, min_level, max_level};
            mure::MRMesh<Config> mesh2{box, min_level, max_level};
           

            auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);
                
    
            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            auto update_bc_for_level2 = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };            

            // Initialization
            auto f   = init_f(mesh);
            auto f2  = init_f(mesh2);

            auto MRadaptation  = mure::make_MRAdapt(f, update_bc_for_level);
            auto MRadaptation2 = mure::make_MRAdapt(f2, update_bc_for_level2);

            double T = 1.3;//0.4;
            double dx = 1.0 / (1 << max_level);
            double dt = dx;

            std::size_t N = static_cast<std::size_t>(T / dt);

            double t = 0.0;

            double regularity = 0.;

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {

                std::cout<<std::endl<<"Iteration = "<<nb_ite<<std::endl;


                save_solution(f, eps, nb_ite, std::string("normal"));
                save_solution(f2, eps, nb_ite, std::string("finest"));
                                                
                MRadaptation( eps, regularity);
                MRadaptation2(eps, regularity);

                save_solution(f, eps, nb_ite, std::string("normal_post"));
                save_solution(f2, eps, nb_ite, std::string("finest_post"));

                one_time_step_overleaves(f, update_bc_for_level, pred_coeff_separate, s, 0);
                one_time_step_overleaves(f2, update_bc_for_level2, pred_coeff_separate, s, 1);

                t += dt;

            }

        }

    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
