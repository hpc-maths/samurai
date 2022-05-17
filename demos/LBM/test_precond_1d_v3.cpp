#include <samurai/mr/mesh.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/algorithm.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/uniform_mesh.hpp>

#include "prediction_map_1d.hpp"

template <class Mesh>
auto init(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, 1>("u", mesh);


    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, auto& i, auto)
    {
        double dx = 1./(1<<level);
        auto x = dx*xt::arange(i.start, i.end) + 0.5*dx;
        // u(level, i) = xt::exp(-100*x*x);

        // u(level, i) = xt::abs(x);
        u(level, i) = xt::tanh(50*xt::abs(x)) - 1;
    });
    return u;
}

int main()
{
    constexpr size_t dim = 1;
    using MRConfig = samurai::MRConfig<dim, 2>;
    using MRMesh = samurai::MRMesh<MRConfig>;
    using mrmesh_id_t = typename MRMesh::mesh_id_t;

    using UConfig = samurai::UniformConfig<dim>;
    using UMesh = samurai::UniformMesh<UConfig>;
    using umesh_id_t = typename UMesh::mesh_id_t;

    std::size_t min_level = 2, max_level = 12;
    samurai::Box<double, dim> box({-1}, {1});
    MRMesh mrmesh {box, min_level, max_level};
    UMesh umesh {box, max_level};
    auto u = init(mrmesh);
    auto u_exact = init(umesh);

    auto uu = samurai::make_field<double, 1>("uu", umesh);
    uu.fill(0.);

    auto error = samurai::make_field<double, 1>("error", umesh);
    error.fill(0.);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
    };

    auto MRadaptation = samurai::make_MRAdapt(u, update_bc_for_level);

    MRadaptation(1e-4, 2);

    auto level_ = samurai::make_field<std::size_t, 1>("level", mrmesh);

    samurai::for_each_cell(mrmesh[mrmesh_id_t::cells], [&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
    });

    // std::size_t delta_level = max_level - min_level;
    // for(int i=0; i < 1<<delta_level; ++i)
    // {
    //     prediction(delta_level, i);
    // }

    samurai::update_ghost_mr(u, update_bc_for_level);
    for(std::size_t level = min_level; level <= max_level; ++level)
    {
        auto set = samurai::intersection(mrmesh[mrmesh_id_t::cells][level], umesh[umesh_id_t::cells])
                 .on(level);
        set([&](auto& i, auto)
        {
            int delta_l = max_level-level;
            if (delta_l == 0)
            {
                uu(level, i) = u(level, i);
            }
            else
            {
                std::size_t nb_cells = 1<<delta_l;
                for (int ii = 0; ii < nb_cells; ++ii)
                {
                    auto pred = prediction(delta_l, ii % nb_cells);

                    for(auto& kv: pred.coeff)
                    {
                        auto i_f = (i<<delta_l) + ii;
                        i_f.step = nb_cells;
                        uu(max_level, i_f) += kv.second*u(level, i + static_cast<int>(kv.first));
                    }
                }
            }
        });
    }

    samurai::for_each_interval(umesh[umesh_id_t::cells], [&](std::size_t level, auto& i, auto)
    {
        error(level, i) = xt::abs(uu(level, i) - u_exact(level, i));
    });

    samurai::save("solution", mrmesh, u, level_);
    samurai::save("solution_uniform", umesh, uu, error);


    // for(int i = 0; i < (1<<level); ++i)
    // {
    //     prediction(level, i);
    // }
    return 0;
}