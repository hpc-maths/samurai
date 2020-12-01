#include <xtensor/xmasked_view.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/cell_flag.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
// #include <samurai/stencil_field.hpp>

#include "stencil_field.hpp"

#include "../FiniteVolume-MR/boundary_conditions.hpp"

#include <chrono>

constexpr size_t dim = 1;

enum class SimpleID
{
    cells = 0,
    cells_and_ghosts = 1,
    count = 2,
    reference = cells_and_ghosts
};

template <>
struct fmt::formatter<SimpleID>: formatter<string_view>
{
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(SimpleID c, FormatContext& ctx) {
    string_view name = "unknown";
    switch (c) {
    case SimpleID::cells:            name = "cells"; break;
    case SimpleID::cells_and_ghosts: name = "cells and ghosts"; break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

template <std::size_t dim_>
struct AMRConfig
{
    static constexpr std::size_t dim = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    static constexpr std::size_t ghost_width = 1;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t = SimpleID;
};

template <class Config>
class AMRMesh: public samurai::Mesh_base<AMRMesh<Config>, Config>
{
public:
    using base_type = samurai::Mesh_base<AMRMesh<Config>, Config>;
    using config = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type = typename base_type::cl_type;
    using lcl_type = typename base_type::lcl_type;

    AMRMesh(const AMRMesh&) = default;
    AMRMesh& operator=(const AMRMesh&) = default;

    AMRMesh(AMRMesh&&) = default;
    AMRMesh& operator=(AMRMesh&&) = default;

    inline AMRMesh(const cl_type &cl, std::size_t min_level, std::size_t max_level)
    : base_type(cl, min_level, max_level)
    {}

    inline AMRMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
    : base_type(b, start_level, min_level, max_level)
    {}

    void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->m_cells[mesh_id_t::cells], [&](std::size_t level, const auto& interval, auto)
        {
            lcl_type& lcl = cl[level];
            samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>([&](auto stencil)
            {
                lcl[{}].add_interval({interval.start - static_cast<int>(config::ghost_width),
                                         interval.end + static_cast<int>(config::ghost_width)});
            });
        });
        this->m_cells[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};



template <class Mesh>
auto init_solution(Mesh & mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        double x = center[0];

        double u = 0.;

        // Initial hat solution
        if (x < -1. or x > 1.)  {
            u = 0.;
        }
        else
        {
            u = (x < 0.) ? (1 + x) : (1 - x);
        }

        // phi[cell] = u;
        phi[cell] = std::exp(-20.*x*x);
    });

    return phi;
}

template<class Field>
inline void amr_projection(Field &field)
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    for (std::size_t level = max_level; level >= min_level; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level],
                                          mesh[mesh_id_t::cells_and_ghosts][level - 1])
                   .on(level - 1);

        expr.apply_op(projection(field));
    }
}

template<class Field, class Func>
inline void amr_prediction(Field &field, Func&& update_bc_for_level)
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh[mesh_id_t::cells].min_level(), max_level = mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level + 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                       samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                                        mesh.get_union()[level]))
                   .on(level);

        expr.apply_op(prediction(field));
        update_bc_for_level(field, level);
    }
}


template <class Field, class Func>
void update_ghosts(Field& phi, Func&& update_bc_for_level)
{
    auto mesh = phi.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    amr_projection(phi);
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        update_bc_for_level(phi, level);
    }
    amr_prediction(phi, std::forward<Func>(update_bc_for_level));
}


template<class Field, class Tag, class GhostUpdate>
void AMR_criterion(Field& f, Tag& tag, GhostUpdate & gu)
{
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep)); // Important

    // We update the values on the ghosts in order
    // to recover good values for the derivatives
    // close the the jumps
    update_ghosts(f, gu);

    for (std::size_t level = min_level; level <= max_level; ++level)    {
        double dx = 1./(1 << level);

        auto leaves = samurai::intersection(mesh[SimpleID::cells][level], 
                                            mesh[SimpleID::cells][level]);

        leaves([&](auto& interval, auto& ) {
            auto k = interval;

            auto der_approx = xt::eval(xt::abs((f(level, k + 1) - f(level, k - 1)) / (2.*dx)));
            auto der_der_approx = xt::eval(xt::abs((f(level, k + 1) - 2.*f(level, k) + f(level, k - 1)) / (dx*dx)));

            auto der_plus  = xt::eval(xt::abs((f(level, k + 1) - f(level, k)) / (dx)));
            auto der_minus = xt::eval(xt::abs((f(level, k) - f(level, k - 1)) / (dx)));

            // auto mask = xt::abs(f(level, k)) > 0.001;
            auto mask = der_approx > 0.01;
            // auto mask = der_der_approx > 0.01;

            // auto mask = (xt::abs(der_plus) - xt::abs(der_minus)) > 0.001;

            if (level == max_level) {
                xt::masked_view(tag(level, k),   mask) = static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, k),  !mask) = static_cast<int>(samurai::CellFlag::coarsen);

            }
            else
            {
                if (level == min_level) {
                    tag(level, k) = static_cast<int>(samurai::CellFlag::keep);
                }
                else
                {
                    xt::masked_view(tag(level, k),   mask) = static_cast<int>(samurai::CellFlag::refine);
                    xt::masked_view(tag(level, k),  !mask) = static_cast<int>(samurai::CellFlag::coarsen);                    
                }   
            }
        });
    }
}


template<class TInterval>
class projection_op_: public samurai::field_operator_base<TInterval>
{
public:
    INIT_OPERATOR(projection_op_)

    template<class T>
    inline void operator()(samurai::Dim<1>,T& new_field, const T& field) const
    {
        new_field(level, i) = .5 * (field(level + 1, 2 * i    )
                                   +field(level + 1, 2 * i + 1));
    }
};

template<class T>
inline auto projection(T&& new_field, T&& field)
{
    return samurai::make_field_operator_function<projection_op_>(std::forward<T>(new_field), std::forward<T>(field));
}

template<class Field>
void make_graduation(Field & tag)
{
    auto mesh = tag.mesh();
    for (std::size_t level = mesh.max_level(); level >= 1; --level)
    {
        // We project the cells at level j back on level j-1
        auto ghost_subset = samurai::intersection(mesh[SimpleID::cells][level],
                                                  mesh[SimpleID::reference][level-1])
                            .on(level - 1);

        ghost_subset([&](const auto& i, const auto& )
        {
            tag(level - 1, i) |= static_cast<int>(samurai::CellFlag::keep);
        });

        auto leaves = intersection(mesh[SimpleID::cells][level],
                                   mesh[SimpleID::cells][level]);

        leaves([&](const auto& i, const auto& )
        {
            xt::xtensor<bool, 1> mask = (tag(level, i) & static_cast<int>(samurai::CellFlag::refine));

            for(int ii = -1; ii <= 1; ++ii)  {
                xt::masked_view(tag(level, i + ii), mask) |= static_cast<int>(samurai::CellFlag::keep);
            }
        });

        leaves.on(level - 1)([&](const auto& i, const auto& )
        {
            xt::xtensor<bool, 1> mask = (  tag(level, 2 * i)     & static_cast<int>(samurai::CellFlag::keep))
                                        | (tag(level, 2 * i + 1) & static_cast<int>(samurai::CellFlag::keep));

            xt::masked_view(tag(level,     2 * i), mask) |= static_cast<int>(samurai::CellFlag::keep);
            xt::masked_view(tag(level, 2 * i + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);

        });

        xt::xtensor_fixed<int, xt::xshape<2, dim>> stencil{{1}, {-1}};

        for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
        {
            auto s = xt::view(stencil, i);
            auto subset = samurai::intersection(samurai::translate(mesh[SimpleID::cells][level], s),
                                             mesh[SimpleID::cells][level - 1])
                         .on(level);

            subset([&](const auto& interval, const auto& )
            {
                auto mask = tag(level, interval  - s[0]) & static_cast<int>(samurai::CellFlag::refine);
                auto half_i = interval >> 1;
                xt::masked_view(tag(level - 1, half_i), mask) |= static_cast<int>(samurai::CellFlag::refine);

                mask = tag(level, interval  - s[0]) & static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level - 1, half_i), mask) |= static_cast<int>(samurai::CellFlag::keep);
            });
        }
    }
}

template<class Field, class Tag>
bool update_mesh(Field& f, const Tag& tag)
{
    using mesh_t = typename Field::mesh_t;
    using interval_t = typename mesh_t::interval_t;
    using coord_index_t = typename interval_t::coord_index_t;
    using cl_type = typename mesh_t::cl_type;

    auto mesh = f.mesh();

    cl_type cell_list;

    samurai::for_each_interval(mesh[SimpleID::cells], [&](std::size_t level, const auto& interval, const auto& )
    {
        for (int i = interval.start; i < interval.end; ++i)
        {
            if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
            {
                cell_list[level + 1][{}].add_interval({2 * i, 2 * i + 2});
            }
            else if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::keep))
            {
                cell_list[level][{}].add_point(i);
            }
            else
            {
                cell_list[level-1][{}].add_point(i>>1);
            }
        }
    });

    mesh_t new_mesh(cell_list, mesh.min_level(), mesh.max_level());

    if (new_mesh == mesh)
    {
        return true;
    }

    Field new_f{f.name(), new_mesh};
    new_f.fill(0.);

    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        auto common_leaves = samurai::intersection(    mesh[SimpleID::cells][level],
                                                new_mesh[SimpleID::cells][level]);

        common_leaves.apply_op(samurai::copy(new_f, f));
    }

    samurai::for_each_interval(mesh[SimpleID::cells], [&](std::size_t level, const auto& interval, const auto& )
    {
        for (coord_index_t i = interval.start; i < interval.end; ++i)
        {
            if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
            {
                samurai::compute_prediction(level, interval_t{i, i + 1}, xt::xtensor_fixed<int,  xt::xshape<0>>{}, f, new_f);
            }
        }
    });

    for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection    (mesh[SimpleID::cells][level],
                                         new_mesh[SimpleID::cells][level - 1])
                     .on(level - 1);
        subset.apply_op(projection(new_f, f));
    }

    f.mesh_ptr()->swap(new_mesh);
    std::swap(f.array(), new_f.array());

    return false;
}


template<class Field>
void save_solution(Field &f, std::size_t ite, std::string ext = "")
{
    // using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "Burgers_AMR" << "_lmin_" << min_level << "_lmax-" << max_level <<"_"<<ext<< "_ite-" << ite;

    auto level_ = samurai::make_field<double, 1>("level", mesh);

    samurai::for_each_cell(mesh[SimpleID::cells], [&](auto &cell)
    {
        level_[cell] = static_cast<double>(cell.level);
    });

    samurai::save(str.str(), mesh, f, level_);

}

template <class Field>
void flux_correction(Field& phi_np1, const Field& phi_n, double dt)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = phi_np1.mesh();
    std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    double dx = 1./(1 << max_level);

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        double dx_loc = 1./(1<<level);
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {{-1}};
        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                               mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_right([&](const auto& i, const auto& )
        {
            phi_np1(level, i) = phi_np1(level, i) + dt/dx_loc * (samurai::upwind_Burgers_op<interval_t>(level, i).right_flux(phi_n, dx/dt)
                                                            -samurai::upwind_Burgers_op<interval_t>(level+1, 2*i+1).right_flux(phi_n, dx/dt));
        });

        stencil = {{1}};
        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                        mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_left([&](const auto& i, const auto& )
        {
            phi_np1(level, i) = phi_np1(level, i) - dt/dx_loc * (samurai::upwind_Burgers_op<interval_t>(level, i).left_flux(phi_n, dx/dt)
                                                            -samurai::upwind_Burgers_op<interval_t>(level+1, 2*i).left_flux(phi_n, dx/dt));
        });
    }
}

int main(int argc, char *argv[])
{
    using Config = AMRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t max_level = 6;
    std::size_t min_level = 1;
    
    samurai::Box<double, dim> box({-3}, {3});
    AMRMesh<Config> mesh{box, max_level, min_level, max_level};


    auto phi = init_solution(mesh);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };


    double Tf = 1.5; // We have blowup at t = 1
    double dx = 1./(1 << max_level);
    double dt = 0.99 * dx; // 0.99 * dx 

    double t = 0.;
    std::size_t it = 0;

    while (t < Tf)  {
        
        std::cout<<std::endl<<"Iteration = "<<it<<"   Time = "<<t<<std::flush;  

        std::size_t ite = 0;
        while(ite < (max_level - min_level + 1))
        {
            std::cout << "Mesh adaptation iteration " << ite++ << std::endl;
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criterion(phi, tag, update_bc_for_level);
            make_graduation(tag);
            if(update_mesh(phi, tag))
            {
                break;
            }
        }    
        
        save_solution(phi, it, "before");

        // Numerical scheme
        update_ghosts(phi, update_bc_for_level);
        auto phinp1 = samurai::make_field<double, 1>("phi", mesh);

        phinp1 = phi - dt * samurai::upwind_Burgers(phi, dx/dt);
        save_solution(phinp1, it, "after");

        flux_correction(phinp1, phi, dt);
        save_solution(phinp1, it, "after_after");

        std::swap(phi.array(), phinp1.array());


        t  += dt;
        it += 1;
    } 



    return 0;
    
    }
