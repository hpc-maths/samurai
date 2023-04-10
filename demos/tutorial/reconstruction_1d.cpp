// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <chrono>
#include <filesystem>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/uniform_mesh.hpp>

namespace fs = std::filesystem;

enum class Case : int
{
    abs,
    exp,
    tanh
};

template <class Mesh>
auto init(Mesh& mesh, Case& c)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& i, const auto&)
                               {
                                   const double dx = samurai::cell_length(level);
                                   auto x          = dx * xt::arange(i.start, i.end) + 0.5 * dx;

                                   switch (c)
                                   {
                                       case Case::abs:
                                           u(level, i) = xt::abs(x);
                                           break;
                                       case Case::exp:
                                           u(level, i) = xt::exp(-100 * x * x);
                                           break;
                                       case Case::tanh:
                                           u(level, i) = xt::tanh(50 * xt::abs(x)) - 1;
                                           break;
                                   }
                               });
    return u;
}

int main(int argc, char* argv[])
{
    constexpr size_t dim                        = 1;
    constexpr std::size_t max_stencil_width_    = 1;
    constexpr std::size_t graduation_width_     = 2;
    constexpr std::size_t max_refinement_level_ = samurai::default_config::max_level;
    constexpr std::size_t prediction_order_     = 1;
    using MRConfig = samurai::MRConfig<dim, max_stencil_width_, graduation_width_, prediction_order_, max_refinement_level_>;

    Case test_case{Case::abs};
    const std::map<std::string, Case> map{
        {"abs",  Case::abs },
        {"exp",  Case::exp },
        {"tanh", Case::tanh}
    };

    // Adaptation parameters
    std::size_t min_level = 3;
    std::size_t max_level = 8;
    double mr_epsilon     = 1.e-4; // Threshold used by multiresolution
    double mr_regularity  = 2.;    // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "reconstruction_1d";

    CLI::App app{"1d reconstruction of an adapted solution using multiresolution"};
    app.add_option("--case", test_case, "Test case")->capture_default_str()->transform(CLI::CheckedTransformer(map, CLI::ignore_case));
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    using MRMesh      = samurai::MRMesh<MRConfig>;
    using mrmesh_id_t = typename MRMesh::mesh_id_t;

    using UConfig = samurai::UniformConfig<dim>;
    using UMesh   = samurai::UniformMesh<UConfig>;

    const samurai::Box<double, dim> box({-1}, {1});
    MRMesh mrmesh{box, min_level, max_level};
    UMesh umesh{box, max_level};
    auto u       = init(mrmesh, test_case);
    auto u_exact = init(umesh, test_case);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    auto level_ = samurai::make_field<std::size_t, 1>("level", mrmesh);
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

    auto error = samurai::make_field<double, 1>("error", u_reconstruct.mesh());
    samurai::for_each_interval(u_reconstruct.mesh(),
                               [&](std::size_t level, const auto& i, const auto&)
                               {
                                   error(level, i) = xt::abs(u_reconstruct(level, i) - u_exact(level, i));
                               });
    samurai::save(path, fmt::format("uniform_{}", filename), u_reconstruct.mesh(), u_reconstruct, error);

    return 0;
}
