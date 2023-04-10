// #include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmasked_view.hpp>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/field_expression.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/operators_base.hpp>
#include <samurai/stencil_field.hpp>

template <class Mesh>
auto init_field(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    auto field      = samurai::make_field<double, 1>("sol", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                               auto x        = cell.center(0);
                               auto y        = cell.center(1);
                               double radius = 0.15;
                               field[cell]   = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.75) * (y - 0.75)) - radius;
                           });
    return field;
}

template <class Mesh>
auto init_velocity(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, 2>("u", mesh);
    u.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                               auto x = cell.center(0);
                               auto y = cell.center(1);

                               u[cell][0] = -std::pow(std::sin(M_PI * x), 2.) * std::sin(2. * M_PI * y);
                               u[cell][1] = std::pow(std::sin(M_PI * y), 2.) * std::sin(2. * M_PI * x);
                           });

    return u;
}

template <class Field>
void update_bc_for_level_(Field& field, std::size_t level)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto mesh       = field.mesh();

    std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    xt::xtensor_fixed<int, xt::xshape<4, 2>> stencils{
        {-1, 0 },
        {1,  0 },
        {0,  -1},
        {0,  1 }
    };
    for (std::size_t is = 0; is < stencils.shape()[0]; ++is)
    {
        auto s            = xt::view(stencils, is);
        std::size_t shift = max_level - level;
        auto set_1        = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), s << shift), mesh.domain()),
                                           mesh[mesh_id_t::reference][level]);

        auto set_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (2 * s) << shift), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            set_1);

        auto set_3 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (3 * s) << shift), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(set_1, set_2));

        set_1.on(level)(
            [&](const auto& i, const auto& index)
            {
                auto j             = index[0];
                field(level, i, j) = field(level, i - s[0], j - s[1]);
            });

        set_2.on(level)(
            [&](const auto& i, const auto& index)
            {
                auto j             = index[0];
                field(level, i, j) = field(level, i - 2 * s[0], j - 2 * s[1]);
            });

        set_3.on(level)(
            [&](const auto& i, const auto& index)
            {
                auto j             = index[0];
                field(level, i, j) = field(level, i - 3 * s[0], j - 3 * s[1]);
            });
    }
}

struct stencil_graduation
{
    static auto call(samurai::Dim<2>)
    {
        return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
            {1,  1 },
            {-1, -1},
            {-1, 1 },
            {1,  -1}
        };
        // return xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{{ 1,  0},
        //                                                         {-1,  0},
        //                                                         { 0,  1},
        //                                                         { 0, -1}};
    }

    static auto call(samurai::Dim<3>)
    {
        return xt::xtensor_fixed<int, xt::xshape<8, 3>>{
            {1,  1,  1 },
            {-1, 1,  1 },
            {1,  -1, 1 },
            {-1, -1, 1 },
            {1,  1,  -1},
            {-1, 1,  -1},
            {1,  -1, -1},
            {-1, -1, -1}
        };
        // return xt::xtensor_fixed<int, xt::xshape<6, 3>> stencil{{ 1,  0,  0},
        //                                                         {-1,  0,  0},
        //                                                         { 0,  1,  0},
        //                                                         { 0, -1,  0},
        //                                                         { 0,  0,  1},
        //                                                         { 0,  0,
        //                                                         -1}};
    }
};

template <class Field, class Tag>
void AMR_criteria(const Field& field, Tag& tag)
{
    using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
    auto mesh             = field.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto cell)
                           {
                               double dx = 1. / (1 << (max_level));

                               if (std::abs(field[cell]) < 1.2 * 5 * std::sqrt(2.) * dx)
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

template <class TInterval>
class os5_op : public samurai::field_operator_base<TInterval>,
               public samurai::field_expression<os5_op<TInterval>>
{
  public:

    INIT_OPERATOR(os5_op)

    template <class View, class Vel>
    auto flux(const View& phi_i, const Vel& vel, const double dt) const
    {
        auto xnu   = xt::abs(vel) * dt / dx();
        auto xcoe2 = 0.5 - xnu * 0.5;
        auto xcoe3 = xcoe2 * (1. + xnu) / 3;
        auto xcoe4 = xcoe3 * (xnu * 0.25 - 0.5);
        auto xcoe5 = xcoe4 * (xnu * 0.2 - 0.6);

        auto coe1 = xcoe5 + xcoe4;
        auto coe2 = xcoe3 - 4. * xcoe4 - 5. * xcoe5;
        auto coe3 = xcoe5 * 10. + 6. * xcoe4 - 3. * xcoe3 + xcoe2 - 1.;
        auto coe4 = 1. - 2. * xcoe2 + 3 * xcoe3 - 4. * xcoe4 - 10. * xcoe5;
        auto coe5 = 5. * xcoe5 + xcoe4 - xcoe3 + xcoe2;

        return xt::eval(coe1 * xt::view(phi_i, 0) + coe2 * xt::view(phi_i, 1) + coe3 * xt::view(phi_i, 2) + coe4 * xt::view(phi_i, 3)
                        + coe5 * xt::view(phi_i, 4) - xcoe5 * xt::view(phi_i, 5));
    }

    template <class Field, class Vel>
    auto flux_x(const Field& phi, const Vel& vel, double dt) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(0, level, i, j) >= 0;

        for (int k = -3, kk = 0; k < 3; ++k, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i + k, j), mask);
        }

        for (int k = 3, kk = 0; k > -3; --k, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i + k, j), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }

        return flux(phi_i, vel(0, level, i, j), dt);
    }

    template <class Field, class Vel>
    auto flux_y(const Field& phi, const Vel& vel, double dt) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(1, level, i, j) >= 0;

        for (int k = -3, kk = 0; k < 3; ++k, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i, j + k), mask);
        }

        for (int k = 3, kk = 0; k > -3; --k, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i, j + k), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }
        return flux(phi_i, vel(1, level, i, j), dt);
    }

    template <class Field, class Vel>
    auto operator()(samurai::Dim<2>, Field& field, const Vel& vel, double dt, std::size_t direction) const
    {
        if (direction == 0)
        {
            return xt::eval(vel(0, level, i, j) * flux_x(field, vel, dt) / dx());
        }
        else
        {
            return xt::eval(vel(1, level, i, j) * flux_y(field, vel, dt) / dx());
        }
    }
};

template <class... CT>
inline auto os5(CT&&... e)
{
    return samurai::make_field_operator_function<os5_op>(std::forward<CT>(e)...);
}

int main()
{
    constexpr std::size_t dim         = 2;
    constexpr std::size_t ghost_width = 3;
    std::size_t start_level           = 6;
    std::size_t min_level             = 2;
    std::size_t max_level             = 8;

    samurai::Box<double, dim> box{
        {0, 0},
        {1, 1}
    };
    using Config = samurai::amr::Config<dim, ghost_width>;
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);

    auto field = init_field(mesh);

    auto field_np1 = samurai::make_field<double, 1>("sol", mesh);
    auto tag       = samurai::make_field<int, 1>("tag", mesh);

    auto update_bc = [](std::size_t level, auto& field)
    {
        update_bc_for_level_(field, level);
    };

    double dt = 0.5 / (1 << max_level);
    double Tf = 2.; // Final time

    std::size_t max_ite = Tf / dt;

    std::array<double, dim> a{1, 0};
    for (std::size_t ite = 0; ite < max_ite; ++ite)
    {
        std::cout << "iteration: " << ite << std::endl;

        std::size_t ite_adapt = 0;
        while (1)
        {
            std::cout << "\tmesh adaptation: " << ite_adapt++ << std::endl;
            samurai::update_ghost(update_bc, field);
            tag.resize();
            AMR_criteria(field, tag);
            samurai::graduation(tag, stencil_graduation::call(samurai::Dim<dim>{}));
            if (samurai::update_field(tag, field))
            {
                break;
            }
        }

        auto vel = init_velocity(mesh);
        field_np1.resize();
        field_np1.fill(0.);

        samurai::update_ghost(update_bc, field);

        if (ite & 1)
        {
            field_np1 = field - dt * os5(field, vel, dt, 1);
            samurai::update_ghost(update_bc, field_np1);
            field_np1 = field_np1 - dt * os5(field_np1, vel, dt, 0);
        }
        else
        {
            field_np1 = field - dt * os5(field, vel, dt, 0);
            samurai::update_ghost(update_bc, field_np1);
            field_np1 = field_np1 - dt * os5(field_np1, vel, dt, 1);
        }

        std::swap(field.array(), field_np1.array());
        samurai::save(fmt::format("os_amr_{}", ite), mesh, field);
    }

    return 0;
}
