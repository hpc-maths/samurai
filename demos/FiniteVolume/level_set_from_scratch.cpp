// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/algorithm/update.hpp>
#include <samurai/algorithm/utils.hpp>
#include <samurai/bc.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/operators.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"

#include <filesystem>
namespace fs = std::filesystem;

enum class SimpleID
{
    cells            = 0,
    cells_and_ghosts = 1,
    count            = 2,
    reference        = cells_and_ghosts
};

template <>
struct fmt::formatter<SimpleID> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(SimpleID c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case SimpleID::cells:
                name = "cells";
                break;
            case SimpleID::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case SimpleID::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

template <std::size_t dim_>
struct AMRConfig
{
    static constexpr std::size_t dim                  = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    static constexpr int max_stencil_width            = 2;
    static constexpr int ghost_width                  = 2;
    static constexpr std::size_t prediction_order     = 1;
    using interval_t                                  = samurai::Interval<int>;
    using mesh_id_t                                   = SimpleID;
};

template <class Config>
class AMRMesh : public samurai::Mesh_base<AMRMesh<Config>, Config>
{
  public:

    using base_type                  = samurai::Mesh_base<AMRMesh<Config>, Config>;
    using config                     = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type   = typename base_type::cl_type;
    using lcl_type  = typename base_type::lcl_type;

    using ca_type = typename base_type::ca_type;

    inline AMRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    inline AMRMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b, start_level, min_level, max_level)
    {
    }

    inline void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(
            this->cells()[mesh_id_t::cells],
            [&](std::size_t level, const auto& interval, const auto& index_yz)
            {
                samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                    [&](auto stencil)
                    {
                        auto index = xt::eval(index_yz + stencil);
                        cl[level][index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
                    });
            });
        this->cells()[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};

template <class TInterval>
class projection_op_ : public samurai::field_operator_base<TInterval>
{
  public:

    INIT_OPERATOR(projection_op_)

    template <class T>
    inline void operator()(samurai::Dim<2>, T& new_field, const T& field) const
    {
        new_field(level, i, j) = .25
                               * (field(level + 1, 2 * i, 2 * j) + field(level + 1, 2 * i, 2 * j + 1) + field(level + 1, 2 * i + 1, 2 * j)
                                  + field(level + 1, 2 * i + 1, 2 * j + 1));
    }
};

template <class T>
inline auto projection(T&& new_field, T&& field)
{
    return samurai::make_field_operator_function<projection_op_>(std::forward<T>(new_field), std::forward<T>(field));
}

template <class Mesh>
auto init_level_set(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center               = cell.center();
                               const double x            = center[0];
                               const double y            = center[1];
                               constexpr double radius   = .15;
                               constexpr double x_center = 0.5;
                               constexpr double y_center = 0.75;

                               phi[cell] = std::sqrt(std::pow(x - x_center, 2.) + std::pow(y - y_center, 2.)) - radius;
                           });

    samurai::make_bc<samurai::Neumann>(phi, 0.);

    return phi;
}

template <class Mesh>
auto init_velocity(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    const double PI = xt::numeric_constants<double>::PI;

    auto u = samurai::make_field<double, 2>("u", mesh);
    u.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                               auto center    = cell.center();
                               const double x = center[0];
                               const double y = center[1];

                               u[cell][0] = -std::pow(std::sin(PI * x), 2.) * std::sin(2. * PI * y);
                               u[cell][1] = std::pow(std::sin(PI * y), 2.) * std::sin(2. * PI * x);
                           });

    samurai::make_bc<samurai::Neumann>(u, 0., 0.);
    // samurai::make_bc<samurai::Dirichlet>(u, [PI](auto& coords)
    // {
    //     return xt::xtensor_fixed<double, xt::xshape<2>>{
    //         -std::pow(std::sin(PI*coords[0]), 2.) *
    //         std::sin(2.*PI*coords[1]),
    //          std::pow(std::sin(PI*coords[1]), 2.) * std::sin(2.*PI*coords[0])
    //     };
    // });

    return u;
}

template <class Field>
void make_graduation(Field& tag)
{
    auto& mesh = tag.mesh();
    for (std::size_t level = mesh.max_level(); level >= 1; --level)
    {
        auto ghost_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::reference][level - 1]).on(level - 1);
        ghost_subset(
            [&](const auto& i, const auto& index)
            {
                auto j = index[0];
                tag(level - 1, i, j) |= static_cast<int>(samurai::CellFlag::keep);
            });

        samurai::for_each_interval(
            mesh[SimpleID::cells][level],
            [&](std::size_t, const auto& i, const auto& index)
            {
                auto j = index[0];
                xt::xtensor<bool, 1> mask = (tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine)); // NOLINT(misc-const-correctness)

                for (int jj = -1; jj < 2; ++jj)
                {
                    for (int ii = -1; ii < 2; ++ii)
                    {
                        xt::masked_view(tag(level, i + ii, j + jj), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }
                }
            });

        auto keep_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]).on(level - 1);
        keep_subset(
            [&](const auto& i, const auto& index)
            {
                auto j = index[0];

                // NOLINTBEGIN(misc-const-correctness)
                xt::xtensor<bool, 1> mask = (tag(level, 2 * i, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep));
                // NOLINTEND(misc-const-correctness)

                xt::masked_view(tag(level, 2 * i, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            });

        xt::xtensor_fixed<int, xt::xshape<4, Field::dim>> stencil{
            {1,  1 },
            {-1, -1},
            {-1, 1 },
            {1,  -1}
        };
        // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0},
        // {0, 1}, {0, -1}};

        for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
        {
            auto s = xt::view(stencil, is);
            auto subset = samurai::intersection(samurai::translate(mesh[SimpleID::cells][level], s), mesh[SimpleID::cells][level - 1]).on(level);

            subset(
                [&](const auto& i, const auto& index)
                {
                    auto i_f = i.even_elements();
                    auto j_f = index[0];

                    if (i_f.is_valid())
                    {
                        auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                        auto i_c  = i_f >> 1;
                        auto j_c  = j_f >> 1;
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                        mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }

                    i_f = i.odd_elements();
                    if (i_f.is_valid())
                    {
                        auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                        auto i_c  = i_f >> 1;
                        auto j_c  = j_f >> 1;
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                        mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }
                });
        }
    }
}

template <class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    auto& mesh            = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    samurai::for_each_cell(mesh[SimpleID::cells],
                           [&](auto cell)
                           {
                               const double dx = 1. / (1 << (max_level));

                               if (std::abs(f[cell]) < 1.2 * 5 * std::sqrt(2.) * dx)
                               {
                                   if (cell.level == max_level)
                                   {
                                       tag[cell] = static_cast<int>(samurai::CellFlag::keep);
                                   }
                                   else
                                   {
                                       tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                                   }
                               }
                               else
                               {
                                   if (cell.level == min_level)
                                   {
                                       tag[cell] = static_cast<int>(samurai::CellFlag::keep);
                                   }
                                   else
                                   {
                                       tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
                                   }
                               }
                           });
}

template <class Field, class Field_u, class Tag>
bool update_mesh(Field& f, Field_u& u, Tag& tag)
{
    constexpr std::size_t dim = Field::dim;
    using mesh_t              = typename Field::mesh_t;
    using cl_type             = typename mesh_t::cl_type;

    auto& mesh = f.mesh();

    cl_type cell_list;

    samurai::for_each_interval(mesh[SimpleID::cells],
                               [&](std::size_t level, const auto& interval, const auto& index_yz)
                               {
                                   auto itag = interval.start + interval.index;
                                   for (int i = interval.start; i < interval.end; ++i)
                                   {
                                       if (tag[itag] & static_cast<int>(samurai::CellFlag::refine))
                                       {
                                           samurai::static_nested_loop<dim - 1, 0, 2>(
                                               [&](auto stencil)
                                               {
                                                   auto index = 2 * index_yz + stencil;
                                                   cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                               });
                                       }
                                       else if (tag[itag] & static_cast<int>(samurai::CellFlag::keep))
                                       {
                                           cell_list[level][index_yz].add_point(i);
                                       }
                                       else
                                       {
                                           cell_list[level - 1][index_yz >> 1].add_point(i >> 1);
                                       }
                                       itag++;
                                   }
                               });

    mesh_t new_mesh(cell_list, mesh.min_level(), mesh.max_level());

    if (new_mesh == mesh)
    {
        return true;
    }

    samurai::update_field(tag, f, u);

    tag.mesh().swap(new_mesh);
    return false;
}

template <class Field>
inline void amr_projection(Field& field)
{
    auto& mesh      = field.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    const std::size_t max_level = mesh.max_level();

    for (std::size_t level = max_level; level >= 1; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells_and_ghosts][level - 1]).on(level - 1);

        expr.apply_op(projection(field));
    }
}

template <class Field>
inline void amr_prediction(Field& field)
{
    auto& mesh      = field.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    const std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    samurai::update_bc(0, field);

    for (std::size_t level = 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                          samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.get_union()[level]))
                        .on(level);

        expr.apply_op(samurai::prediction<1, false>(field));
        samurai::update_bc(level, field);
    }
}

template <class Field, class Field_u>
void update_ghosts(Field& phi, Field_u& u)
{
    amr_projection(phi);
    amr_projection(u);

    amr_prediction(phi);
    amr_prediction(u);
}

template <class Field, class Field_u>
void flux_correction(Field& phi_np1, const Field& phi_n, const Field_u& u, double dt)
{
    constexpr std::size_t dim = Field::dim;
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;

    auto& mesh                  = phi_np1.mesh();
    const std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    const std::size_t max_level = mesh[mesh_id_t::cells].max_level();
    for (std::size_t level = min_level; level < max_level; ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           + dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).right_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(u, phi_n, dt));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           - dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).left_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j).left_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(u, phi_n, dt));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           + dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).up_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(u, phi_n, dt));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           - dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).down_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j).down_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(u, phi_n, dt));
            });
    }
}

template <class Field, class Phi>
void save(const fs::path& path, const std::string& filename, const Field& u, const Phi& phi, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, phi, u, level_);
}

int main(int argc, char* argv[])
{
    constexpr size_t dim = 2;
    using Config         = AMRConfig<dim>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    double Tf                                             = 3.14;
    double cfl                                            = 5. / 8;

    // AMR parameters
    std::size_t start_level = 8;
    std::size_t min_level   = 4;
    std::size_t max_level   = 8;
    bool correction         = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_level_set_2d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example with a level set in 2d using AMR"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--start-level", start_level, "Start level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_option("--min-level", min_level, "Minimum level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_option("--max-level", max_level, "Maximum level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_flag("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("AMR parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    AMRMesh<Config> mesh{box, max_level, min_level, max_level};

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    // We initialize the level set function
    // We initialize the velocity field

    auto phi    = init_level_set(mesh);
    auto phinp1 = samurai::make_field<double, 1>("phi", mesh);

    auto u = init_velocity(mesh);

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        // AMR adaptation
        std::size_t ite = 0;
        while (true)
        {
            std::cout << "Mesh adaptation iteration " << ite++ << std::endl;
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criteria(phi, tag);
            make_graduation(tag);
            update_ghosts(phi, u);

            if (update_mesh(phi, u, tag))
            {
                break;
            }
        }

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        // Numerical scheme
        update_ghosts(phi, u);
        phinp1.resize();
        phinp1 = phi - dt * samurai::upwind_variable(u, phi, dt);
        if (correction)
        {
            flux_correction(phinp1, phi, u, dt);
        }

        std::swap(phi.array(), phinp1.array());

        // Reinitialization of the level set
        const std::size_t fict_iteration = 2;         // Number of fictitious iterations
        const double dt_fict             = 0.01 * dt; // Fictitious Time step

        auto phi_0 = phi;
        for (std::size_t k = 0; k < fict_iteration; ++k)
        {
            // //Forward Euler - OK
            // update_ghosts(phi, u);
            // phinp1 = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            // std::swap(phi.array(), phinp1.array());

            // TVD-RK2
            update_ghosts(phi, u);
            auto phihat = samurai::make_field<double, 1>("phi", mesh);
            samurai::make_bc<samurai::Neumann>(phihat, 0.);
            phihat = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            update_ghosts(phihat, u);
            phinp1 = .5 * phi_0 + .5 * (phihat - dt_fict * H_wrap(phihat, phi_0, max_level));
            std::swap(phi.array(), phinp1.array());
        }

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, phi, suffix);
        }
    }

    return 0;
}
