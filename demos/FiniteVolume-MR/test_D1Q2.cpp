#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <xtensor/xio.hpp>

#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/harten.hpp>
#include <samurai/mr/adapt.hpp>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"

#include <chrono>


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


// To decide which test we want to perform.

/*

1 : transport - gaussienne
2 : transport - probleme de Riemann
3 : Burgers - tangente hyperbolique reguliere
4 : Burgers - fonction chapeau avec changement de regularite
5 : Burgers - probleme de Riemann
*/

const int test_number = 4   ;

const double ad_vel = 0.75; // Should be < lambda




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

double exact_solution(double x, double t)   {

    double u = 0;


    switch(test_number) {
        case 1 : {
            u = exp(-20.0 * (x-ad_vel*t) * (x-ad_vel*t)); // Used in the first draft
            // u = exp(-60.0 * (x-ad_vel*t) * (x-ad_vel*t));

            break;
        }

        case 2 : {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            double xtr = x - ad_vel*t;
            u =  (xtr <= -sigma) ? (rhoL) : ((xtr <= sigma) ? (rhoC) : rhoR );
            break;
        }
        case 3 : {
            double sigma = 100.0;
            if (t <= 0.0)
                u = 0.5 * (1.0 + tanh(sigma * x));
            else
            {   // We proceed by dicothomy
                double a = -3.2;
                double b =  3.2;

                double tol = 1.0e-8;

                auto F = [sigma, x, t] (double y)   {
                    return y + 0.5 * (1.0 + tanh(sigma * y))*t - x;
                };
                double res = 0.0;

                while (b-a > tol)   {
                    double mean = 0.5 * (b + a);
                    double eval = F(mean);
                    if (eval <= 0.0)
                        a = mean;
                    else
                        b = mean;
                    res = mean;
                }

                u =  0.5 * (1.0 + tanh(sigma * res));
            }
            break;
        }

        case 4 : {
            if (x >= -1 and x < t)
            {
                u = (1 + x) / (1 + t);
            }

            if (x >= t and x < 1)
            {
                u = (1 - x) / (1 - t);
            }
            break;
        }

        case 5 : {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            u =  (x + sigma <= rhoL * t) ? rhoL : ((x + sigma <= rhoC*t) ? (x+sigma)/t : ((x-sigma <= t/2*(rhoC + rhoR)) ? rhoC : rhoR ));
            break;
        }
    }

    return u;
}

double flux(double u)   {

    if (test_number == 1 or test_number == 2)   {
    // if (test_number == 1 or test_number == 2 or test_number == 4)   {

        return ad_vel * u;
    }
    else
    {
        return 0.5 * u *u;
    }


}

template<class Config>
auto init_f(samurai::MRMesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 2;
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto corner = cell.corner();
        double dx = cell.length;

        auto x = corner[0] + .5*dx;
        double u = 0;

        u = exact_solution(x, 0.0);
        double v = flux(u);

        f[cell][0] = .5 * (u + v);
        f[cell][1] = .5 * (u - v);
    });

    return f;
}

template<class Field, class interval_t, class FieldTag>
xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item,
                                  const FieldTag & tag, std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>,
                                  xt::xtensor<double, 1>> & mem_map)
{

    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i});
    if (it != mem_map.end())   {
        //std::cout<<std::endl<<"Found by memoization";
        return it->second;
    }
    else {

        auto mesh = f.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});//xt::eval(f(item, level_g, i));
        auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, i);

        // std::cout << level_g + level << " " << i << " " << mask << "\n";
        if (xt::all(mask))
        {
            return xt::eval(f(item, level_g + level, i));
        }

        auto step = i.step;
        auto ig = i / 2;
        ig.step = step >> 1;
        xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

        for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
        {
            d[iii] = (ii & 1)? -1.: 1.;
        }


        auto val = xt::eval(prediction(f, level_g, level-1, ig, item, tag, mem_map) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, tag, mem_map)
                                                                                       - prediction(f, level_g, level-1, ig-1, item, tag, mem_map)));


        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
            }
        }

        // The value should be added to the memoization map before returning
        return mem_map[{item, level_g, level, i}] = out;

        //return out;
    }

}


// template<class Field, class interval_t>
// xt::xtensor<double, 2> prediction_all(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i,
//                                   std::map<std::tuple<std::size_t, std::size_t, interval_t>,
//                                   xt::xtensor<double, 2>> & mem_map)
// {

//     using namespace xt::placeholders;
//     // We check if the element is already in the map
//     auto it = mem_map.find({level_g, level, i});
//     if (it != mem_map.end())
//     {
//         return it->second;
//     }
//     else
//     {
//         auto mesh = f.mesh();
//         std::vector<std::size_t> shape = {i.size(), 2};
//         xt::xtensor<double, 2> out = xt::empty<double>(shape);
//         auto mask = mesh.exists(mesh_id_t::cells, level_g + level, i);

//         xt::xtensor<double, 2> mask_all = xt::empty<double>(shape);
//         xt::view(mask_all, xt::all(), 0) = mask;
//         xt::view(mask_all, xt::all(), 1) = mask;

//         if (xt::all(mask))
//         {
//             return xt::eval(f(level_g + level, i));
//         }

//         auto ig = i >> 1;
//         ig.step = 1;

//         xt::xtensor<double, 2> val = xt::empty<double>(shape);
//         auto current = xt::eval(prediction_all(f, level_g, level-1, ig, mem_map));
//         auto left = xt::eval(prediction_all(f, level_g, level-1, ig-1, mem_map));
//         auto right = xt::eval(prediction_all(f, level_g, level-1, ig+1, mem_map));

//         std::size_t start_even = (i.start&1)? 1: 0;
//         std::size_t start_odd = (i.start&1)? 0: 1;
//         std::size_t end_even = (i.end&1)? ig.size(): ig.size()-1;
//         std::size_t end_odd = (i.end&1)? ig.size()-1: ig.size();
//         xt::view(val, xt::range(start_even, _, 2)) = xt::view(current - 1./8 * (right - left), xt::range(start_even, _));
//         xt::view(val, xt::range(start_odd, _, 2)) = xt::view(current + 1./8 * (right - left), xt::range(_, end_odd));

//         xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);
//         for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, ++i_int)
//         {
//             if (mask[i_mask])
//             {
//                 xt::view(out, i_mask) = xt::view(f(level_g + level, {i_int, i_int + 1}), 0);
//             }
//         }

//         // The value should be added to the memoization map before returning
//         return out;// mem_map[{level_g, level, i, ig}] = out;
//     }
// }



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


template<class Field, class Func, class FieldTag>
void one_time_step(Field &f, Func&& update_bc_for_level, const FieldTag & tag, double s)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 1.;//, s = 1.0;
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    using interval_t = typename decltype(mesh)::interval_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    samurai::mr_projection(f);
    for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    samurai::mr_prediction(f, update_bc_for_level);


    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

    auto new_f = samurai::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level],
                                      mesh[mesh_id_t::cells][level]);
        exp([&](auto &interval, auto) {
            auto i = interval;


            // STREAM

            std::size_t j = max_level - level;

            double coeff = 1. / (1 << j);

            // This is the STANDARD FLUX EVALUATION

            auto fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, tag, memoization_map));

            auto fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, tag, memoization_map));



            // COLLISION

            auto uu = xt::eval(fp + fm);
            auto vv = xt::eval(lambda * (fp - fm));

            if (test_number == 1 or test_number == 2)   {
            // if (test_number == 1 or test_number == 2 or test_number == 4)   {

                vv = (1 - s) * vv + s * ad_vel * uu;
            }
            else
            {
                vv = (1 - s) * vv + s * .5 * uu * uu;
            }

            new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
        });
    }

    std::swap(f.array(), new_f.array());
}



template<class Field, class Func, class Pred>
void one_time_step_matrix_overleaves(Field &f, Func&& update_bc_for_level, 
                            const Pred& pred_coeff, double s_rel, 
                            bool finest_collision = false)
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

    samurai::mr_projection(f);
    for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    samurai::mr_prediction(f, update_bc_for_level);


    // After that everything is ready, we predict what is remaining
    samurai::mr_prediction_overleaves(f, update_bc_for_level);

    auto new_f = samurai::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);

    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);

    auto help_f = samurai::make_field<double, nvel>("help_f", mesh);
    help_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        // If we are at the finest level, we no not need to correct

        if (level == max_level) {
            std::size_t j = 0;
            double coeff = 1.;


            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level],
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
            // auto overleaves = samurai::intersection(mesh[mesh_id_t::cells][level],
            //                                      mesh[mesh_id_t::cells][level]).on(level + 1);

            auto ol = samurai::intersection(mesh[mesh_id_t::cells][level],
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

                // auto uu = xt::eval(fp + fm);
                // auto vv = xt::eval(lambda*(fp - fm));

                // vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                // help_f(0, level + 1, k) = .5 * (uu + 1. / lambda * vv);
                // help_f(1, level + 1, k) = .5 * (uu - 1. / lambda * vv);

            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                             mesh[mesh_id_t::cells][level]);

            leaves([&](auto &interval, auto) {
                auto k = interval;

                // Projection
                advected_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1)));
                advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1)));
                // new_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1)));
                // new_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1)));
            });
        }
    }


    // Collision


    if (!finest_collision)  {
        // for (std::size_t level = max_level; level <= max_level; ++level)    {

        for (std::size_t level = 0; level <= max_level; ++level)    {

            double dx = 1./(1 << level);

            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
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

            

                    auto pol_at_centers = (.5/(dx*dx)*(uum1+uup1)-1./(dx*dx)*uu) * xt::pow(cc, 2.)
                                + (-(.5*dx+cc)/(dx*dx)*uum1+2.*cc/(dx*dx)*uu+(.5*dx-cc)/(dx*dx)*uup1) * cc
                                + ((-1./24+.5*xt::pow(cc, 2.)/(dx*dx))*(uum1+uup1)+(.5*cc/dx)*(uum1-uup1)+(13./12-xt::pow(cc, 2.)/(dx*dx))*uu);

                    vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(pol_at_centers, 2.);

                }
                else
                {
                    vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(uu, 2.);

                }
                
                // if (test_number == 1 or test_number == 2)   {
                
                //     vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
                // }
                // else
                // {
                //     vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;
                // }


                // vv = (1 - s_rel) * vv + s_rel * .5 * xt::pow(cc, 2.);
                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);

            });
        }
    }

    else {

        samurai::mr_projection(advected_f);
        for (std::size_t level = mesh.min_level() - 1; level <= mesh.max_level(); ++level)
        {
            update_bc_for_level(advected_f, level); 
        }
        samurai::mr_prediction(advected_f, update_bc_for_level);


            
        std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                        xt::xtensor<double, 2>> memoization_map;

        for (std::size_t level = 0; level <= max_level; ++level)    {
                
                
                
            // std::cout<<std::endl<<"Level = "<<level<<std::endl;


            auto leaves_on_finest = samurai::intersection(mesh[mesh_id_t::cells][level],
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

                if (test_number == 1 or test_number == 2)   {
                
                    vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
                }
                else
                {
                    vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;
                }

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


// template<class Field, class FieldR>
template<class Config, class FieldR, class Func>
std::array<double, 2> compute_error(samurai::Field<Config, double, 2> &f, FieldR & fR, Func&& update_bc_for_level, double t)
{

    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    update_bc_for_level(fR, max_level); // It is important to do so

    samurai::mr_projection(f);
    for (std::size_t level = mesh.min_level() - 1; level <= mesh.max_level(); ++level)
    {
        update_bc_for_level(f, level); // It is important to do so
    }
    samurai::mr_prediction(f, update_bc_for_level);

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;

    error_memoization_map.clear();

    double error = 0; // To return
    double diff = 0.0;


    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level],
                                      mesh[mesh_id_t::cells][level])
                  .on(max_level);

        exp([&](auto &interval, auto) {
            auto i = interval;
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));


            xt::xtensor<double, 1> x = dx*xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5*dx;
            // xt::xtensor<double, 1> uexact = (x >= -1.0 and x < t) * ((1 + x) / (1 + t)) +
            //                                 (x >= t and x < 1) * (1 - x) / (1 - t);

            xt::xtensor<double, 1> uexact = xt::zeros<double>(x.shape());

            for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)    {
                uexact[idx] = exact_solution(x[idx], t); // We can probably do better
            }

            auto rho_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i));
            auto rho = xt::eval(xt::view(sol, xt::all(), 0) +  xt::view(sol, xt::all(), 1));

            error += xt::sum(xt::abs(rho_ref - uexact))[0];
            diff  += xt::sum(xt::abs(rho_ref - rho))[0];

            // error += xt::sum(xt::abs(xt::flatten(xt::view(fR(max_level, i), xt::all(), xt::range(0, 1)) + xt::view(fR(max_level, i), xt::all(), xt::range(1, 2)))
            //                  - uexact))[0];


            // diff += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), xt::range(0, 1)) + xt::view(sol, xt::all(), xt::range(1, 2))) - xt::flatten(xt::view(fR(max_level, i), xt::all(), xt::range(0, 1)) + xt::view(fR(max_level, i), xt::all(), xt::range(1, 2)))))[0];


        });
    }

    return {dx * error, dx * diff}; // Normalization by dx before returning
    // I think it is better to do the normalization at the very end ... especially for round-offs
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
            using Config = samurai::MRConfig<dim, 2>;
            using mesh_t = samurai::MRMesh<Config>;
            using mesh_id_t = typename mesh_t::mesh_id_t;
            using coord_index_t = typename mesh_t::interval_t::coord_index_t;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = 2;//result["min_level"].as<std::size_t>();
            std::size_t max_level = 9;//result["max_level"].as<std::size_t>();


            // We set some parameters according
            // to the problem.
            double sol_reg = 0.0;
            double T = 0.0;
            std::string case_name;

            switch(test_number){
                case 1 : {
                    sol_reg = 600.0; // The solution is very smooth
                    // sol_reg = 1.0; // The solution is very smooth
                    T = 0.4;
                    case_name = std::string("t_r");
                    break;
                }
                case 2 : {
                    sol_reg = 0.0;
                    T = 0.4;
                    case_name = std::string("t_d");
                    break;
                }
                case 3 : {
                    sol_reg = 600.0;
                    // sol_reg = 1.0;
                    T = 0.4;
                    case_name = std::string("b_r");
                    break;
                }
                case 4 : {
                    sol_reg = 0.0;
                    T = 1.3; // Let it develop the discontinuity
                    // T = 0.1; // CHANGE

                    case_name = std::string("b_c");
                    break;
                }
                case 5 : {
                    sol_reg = 0.0;
                    T = 0.7;
                    case_name = std::string("b_d");
                    break;
                }
            }

            samurai::Box<double, dim> box({-3}, {3});

            std::vector<double> s_vect {0.75, 1.0, 1.25, 1.5, 1.75};

            auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            for (auto s : s_vect)   {
                std::cout<<std::endl<<"Relaxation parameter s = "<<s;

                std::string prefix (case_name + "_s_"+std::to_string(s)+"_");

                std::cout<<std::endl<<"Testing time behavior"<<std::endl;
                {
                    double eps = 1.0e-4; // This remains fixed

                    samurai::MRMesh<Config> mesh{box, min_level, max_level};
                    samurai::MRMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme


                    // Initialization
                    auto f  = init_f(mesh , 0.0);
                    auto f_old  = init_f(mesh , 0.0);
                    auto fR = init_f(meshR, 0.0);

                    double dx = 1.0 / (1 << max_level);
                    double dt = dx; // Since lb = 1

                    std::size_t N = static_cast<std::size_t>(T / dt);

                    double t = 0.0;

                    std::ofstream out_time_frames;
                    std::ofstream out_error_exact_ref;
                    std::ofstream out_diff_ref_adap;
                    std::ofstream out_compression;

                    out_time_frames.open     ("./d1q2/time/"+prefix+"time.dat");
                    out_error_exact_ref.open ("./d1q2/time/"+prefix+"error.dat");
                    out_diff_ref_adap.open   ("./d1q2/time/"+prefix+"diff.dat");
                    out_compression.open     ("./d1q2/time/"+prefix+"comp.dat");

                    auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                    for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                    {
                        // for (std::size_t i=0; i<max_level-min_level; ++i)
                        // {
                        //     if (coarsening(f, eps, i))
                        //         break;
                        // }

                        // for (std::size_t i=0; i<max_level-min_level; ++i)
                        // {
                        //     if (refinement(f, eps, sol_reg, i))
                        //         break;
                        // }
                        MRadaptation(eps, sol_reg);

                        // auto mesh_old = mesh;
                        // auto f_old = samurai::make_field<double , 2>("u", mesh_old);
                        // f_old.array() = f.array();
                        // for (std::size_t i=0; i<max_level-min_level; ++i)
                        // {
                        //     if (harten(f, f_old, eps, sol_reg, i, nb_ite))
                        //         break;
                        // }

                        // samurai::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                        // tag_leaf.array().fill(0);
                        // mesh.for_each_cell([&](auto &cell) {
                        //     tag_leaf[cell] = static_cast<int>(1);
                        // });

                        // samurai::Field<Config, int, 1> tag_leafR{"tag_leafR", meshR};
                        // tag_leafR.array().fill(0);
                        // meshR.for_each_cell([&](auto &cell) {
                        //     tag_leafR[cell] = static_cast<int>(1);
                        // });

                        auto error = compute_error(f, fR, update_bc_for_level, t);

                        out_time_frames    <<t       <<std::endl;
                        out_error_exact_ref<<error[0]<<std::endl;
                        out_diff_ref_adap  <<error[1]<<std::endl;
                        out_compression    <<static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                           / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))<<std::endl;

                        std::cout<<std::endl<<"n = "<<nb_ite<<"   Time = "<<t<<" Diff = "<<error[1];


                        // one_time_step(f, tag_leaf, s);
                        one_time_step_matrix_overleaves(f, update_bc_for_level, pred_coeff_separate, s, false);
                        one_time_step_matrix_overleaves(fR, update_bc_for_level, pred_coeff_separate, s);

                        // one_time_step(fR, tag_leafR, s);

                        t += dt;

                    }

                    std::cout<<std::endl;

                    out_time_frames.close();
                    out_error_exact_ref.close();
                    out_diff_ref_adap.close();
                    out_compression.close();
                }

                std::cout<<std::endl<<"Testing eps behavior"<<std::endl;
                {
                    double eps = 0.1;
                    std::size_t N_test = 50;
                    double factor = 0.60;
                    std::ofstream out_eps;
                    std::ofstream out_diff_ref_adap;
                    std::ofstream out_compression;
                    std::ofstream out_max_level;


                    out_eps.open             ("./d1q2/eps/"+prefix+"eps.dat");
                    out_diff_ref_adap.open   ("./d1q2/eps/"+prefix+"diff.dat");
                    out_compression.open     ("./d1q2/eps/"+prefix+"comp.dat");
                    out_max_level.open       ("./d1q2/eps/"+prefix+"maxlevel.dat");

                    for (std::size_t n_test = 0; n_test < N_test; ++ n_test)    {
                        std::cout<<std::endl<<"Test "<<n_test<<" eps = "<<eps;

                        mesh_t mesh{box, min_level, max_level};
                        mesh_t meshR{box, max_level, max_level}; // This is the reference scheme

                        // Initialization
                        auto f  = init_f(mesh , 0.0);
                        auto f_old  = init_f(mesh , 0.0);
                        auto fR = init_f(meshR, 0.0);


                        double dx = 1.0 / (1 << max_level);
                        double dt = dx; // Since lb = 1

                        std::size_t N = static_cast<std::size_t>(T / dt);

                        double t = 0.0;


                        double foo_diff = 0.0;

                        auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                        // for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)

                        {

                            MRadaptation(eps, sol_reg);


                            one_time_step_matrix_overleaves(f, update_bc_for_level, pred_coeff_separate, s, false);
                            one_time_step_matrix_overleaves(fR, update_bc_for_level, pred_coeff_separate, s);


                            t += dt;

                        }


                        auto error = compute_error(f, fR, update_bc_for_level, t);
                        std::cout<<"Diff = "<<error[1]<<std::endl;


                        std::size_t max_level_effective = mesh.min_level();

                        for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)  {


                            if (!mesh[mesh_id_t::cells][level].empty())
                                max_level_effective = level;

                        }

                        out_max_level<<max_level_effective<<std::endl;


                        out_eps<<eps<<std::endl;
                        out_diff_ref_adap<<error[1]<<std::endl;
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
    }

    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
