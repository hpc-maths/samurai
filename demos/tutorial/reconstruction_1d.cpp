#include <chrono>

#include <samurai/mr/mesh.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/algorithm.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/uniform_mesh.hpp>

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

        u(level, i) = xt::abs(x);
        // u(level, i) = xt::tanh(50*xt::abs(x)) - 1;
    });
    return u;
}

int main()
{
    constexpr size_t dim = 1;
    constexpr std::size_t max_stencil_width_ = 1;
    constexpr std::size_t graduation_width_ = 2;
    constexpr std::size_t max_refinement_level_ = samurai::default_config::max_level;
    constexpr std::size_t prediction_order_ = 1;
    using MRConfig = samurai::MRConfig<dim, max_stencil_width_,
        graduation_width_,
        max_refinement_level_,
        prediction_order_
    >;
    using MRMesh = samurai::MRMesh<MRConfig>;
    using mrmesh_id_t = typename MRMesh::mesh_id_t;

    using UConfig = samurai::UniformConfig<dim>;
    using UMesh = samurai::UniformMesh<UConfig>;
    using umesh_id_t = typename UMesh::mesh_id_t;

    std::size_t min_level = 3, max_level = 10;
    samurai::Box<double, dim> box({-1}, {1});
    MRMesh mrmesh {box, min_level, max_level};
    UMesh umesh {box, max_level};
    auto u = init(mrmesh);
    auto u_exact = init(umesh);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
    };

    auto MRadaptation = samurai::make_MRAdapt(u, update_bc_for_level);

    MRadaptation(1e-4, 2);

    auto level_ = samurai::make_field<std::size_t, 1>("level", mrmesh);

    samurai::for_each_cell(mrmesh[mrmesh_id_t::cells], [&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
    });

    samurai::update_ghost_mr(u, update_bc_for_level);

    samurai::save("solution", mrmesh, u, level_);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto u_reconstruct = reconstruction(u);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "execution time " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

    auto error = samurai::make_field<double, 1>("error", u_reconstruct.mesh());
    samurai::for_each_interval(u_reconstruct.mesh(), [&](std::size_t level, auto& i, auto)
    {
        error(level, i) = xt::abs(u_reconstruct(level, i) - u_exact(level, i));
    });

    samurai::save("solution_uniform", u_reconstruct.mesh(), u_reconstruct, error);
    return 0;
}