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

        phi[cell] = u;
    });

    return phi;
}


template<class Field>
void save_solution(Field &f, std::size_t ite, std::string ext = "")
{
    // using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "Burgers_AMR" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_ite-" << ite;

    // auto h5file = samurai::Hdf5(str.str().data());
    // h5file.add_mesh(mesh);

    auto level_ = samurai::make_field<double, 1>("level", mesh);

    samurai::for_each_cell(mesh[SimpleID::cells], [&](auto &cell)
    {
        level_[cell] = static_cast<double>(cell.level);
    });

    samurai::save(str.str(), mesh, f, level_);

    // h5file.add_field(f);
    // h5file.add_field(level_);
}

int main(int argc, char *argv[])
{
    using Config = AMRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t max_level = 8;
    std::size_t min_level = max_level;
    
    samurai::Box<double, dim> box({-3}, {3});
    AMRMesh<Config> mesh{box, max_level, min_level, max_level};


    std::cout<<std::endl<<mesh<<std::endl;


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
        
        save_solution(phi, it);

        // Numerical scheme

        auto phinp1 = samurai::make_field<double, 1>("phi", mesh);

        phinp1 = phi - dt * samurai::upwind_Burgers(phi, dx/dt);

        std::swap(phi.array(), phinp1.array());


        t  += dt;
        it += 1;
    } 




    return 0;
    
    }
