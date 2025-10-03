// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <chrono>
#include <filesystem>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>
#include <samurai/uniform_mesh.hpp>

namespace fs = std::filesystem;

enum class Case : int
{
    abs,
    exp,
    tanh
};

using namespace samurai::math;

template <class Mesh>
auto init(Mesh& mesh, Case& c)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& i, const auto& index)
                               {
                                   auto j          = index[0];
                                   const double dx = mesh.cell_length(level);
                                   auto x          = mesh.origin_point()[0] + dx * arange<double>(i.start, i.end) + 0.5 * dx;
                                   auto y          = mesh.origin_point()[1] + j * dx + 0.5 * dx;

                                   switch (c)
                                   {
                                       case Case::abs:
                                           u(level, i, j) = abs(x) + std::abs(y);
                                           break;
                                       case Case::exp:
                                           u(level, i, j) = exp(-100 * (x * x + y * y));
                                           break;
                                       case Case::tanh:
                                           u(level, i, j) = tanh(50 * (abs(x) + std::abs(y))) - 1;
                                           break;
                                   }
                               });

    switch (c)
    {
        case Case::abs:
            samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                    [](auto, auto, const auto& coords)
                                                    {
                                                        return std::abs(coords[0]) + std::abs(coords[1]);
                                                    });
            break;
        case Case::exp:
            samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                    [](auto, auto, const auto& coords)
                                                    {
                                                        return std::exp(-100 * (coords[0] * coords[0] + coords[1] * coords[1]));
                                                    });
            break;
        case Case::tanh:
            samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                    [](auto, auto, const auto& coords)
                                                    {
                                                        return std::tanh(50 * (std::abs(coords[0]) + std::abs(coords[1]))) - 1;
                                                    });
            break;
    }

    return u;
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("2d reconstruction of an adapted solution using multiresolution", argc, argv);

    constexpr size_t dim = 2;

    Case test_case{Case::abs};
    const std::map<std::string, Case> map{
        {"abs",  Case::abs },
        {"exp",  Case::exp },
        {"tanh", Case::tanh}
    };

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "reconstruction_2d";

    app.add_option("--case", test_case, "Test case")->capture_default_str()->transform(CLI::CheckedTransformer(map, CLI::ignore_case));
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    using UConfig = samurai::UniformConfig<dim>;
    using UMesh   = samurai::UniformMesh<UConfig>;

    const samurai::Box<double, dim> box({-1, -1}, {1, 1});
    // clang-format off
    auto config = samurai::mesh_config<dim>()
    .min_level(3)
    .max_level(8)
    .scaling_factor(1)
    .graduation_width(2)
    .max_stencil_radius(2);
    // clang-format on
    auto mrmesh = samurai::make_MRMesh(config, box);
    UMesh umesh{box, mrmesh.max_level(), 0, 1};
    auto u       = init(mrmesh, test_case);
    auto u_exact = init(umesh, test_case);

    using mrmesh_id_t = typename decltype(mrmesh)::mesh_id_t;

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().regularity(2);
    MRadaptation(mra_config);

    auto level_ = samurai::make_scalar_field<std::size_t>("level", mrmesh);
    samurai::for_each_cell(mrmesh[mrmesh_id_t::cells],
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });
    samurai::save(path, filename, mrmesh, u, level_);

    auto t1            = std::chrono::high_resolution_clock::now();
    auto u_reconstruct = reconstruction(u);
    auto t2            = std::chrono::high_resolution_clock::now();
    std::cout << "execution time " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

    auto error = samurai::make_scalar_field<double>("error", u_reconstruct.mesh());
    samurai::for_each_interval(u_reconstruct.mesh(),
                               [&](std::size_t level, const auto& i, const auto& index)
                               {
                                   auto j             = index[0];
                                   error(level, i, j) = abs(u_reconstruct(level, i, j) - u_exact(level, i, j));
                               });
    samurai::save(path, fmt::format("uniform_{}", filename), u_reconstruct.mesh(), u_reconstruct, error);

    samurai::finalize();
    return 0;
}
