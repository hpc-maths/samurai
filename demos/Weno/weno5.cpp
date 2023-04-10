#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mesh.hpp>
#include <samurai/static_algorithm.hpp>

#include <samurai/field_expression.hpp>
#include <samurai/operators_base.hpp>

#include <fmt/format.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmasked_view.hpp>

enum class AMR_Id
{
    cells            = 0,
    cells_and_ghosts = 1,
    count            = 2,
    reference        = cells_and_ghosts
};

template <>
struct fmt::formatter<AMR_Id> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(AMR_Id c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case AMR_Id::cells:
                name = "cells";
                break;
            case AMR_Id::cells_and_ghosts:
                name = "cells and ghosts";
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
    static constexpr int ghost_width                  = 3;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t  = AMR_Id;
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
        for_each_interval(this->m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cl[level];
                              samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
                                  });
                          });
        this->m_cells[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};

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
void update_bc(Field& field)
{
    using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
    auto mesh             = field.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    xt::xtensor_fixed<int, xt::xshape<4, 2>> stencils{
        {-3, 0 },
        {3,  0 },
        {0,  -3},
        {0,  3 }
    };
    for (std::size_t level = min_level; level < max_level; ++level)
    {
        for (std::size_t is = 0; is < stencils.shape()[0]; ++is)
        {
            auto s   = xt::view(stencils, is);
            auto set = samurai::difference(samurai::translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level]);

            set(
                [&](const auto& i, const auto& index)
                {
                    auto j             = index[0];
                    field(level, i, j) = field(level, i - s[0], j - s[1]);
                });
        }
    }
}

template <class TInterval>
class weno5_op : public samurai::field_operator_base<TInterval>,
                 public samurai::field_expression<weno5_op<TInterval>>
{
  public:

    INIT_OPERATOR(weno5_op)

    template <class T, class Field, class Vel>
    inline auto vel_x_pos(T& dphi, const Field& phi, const Vel& vel) const
    {
        double inv_dx = 1. / dx();
        double eps    = 1e-6;
        auto mask     = vel(0, level, i, j) >= 0;

        auto phi_m3 = xt::eval(xt::masked_view(phi(level, i - 3, j), mask));
        auto phi_m2 = xt::eval(xt::masked_view(phi(level, i - 2, j), mask));
        auto phi_m1 = xt::eval(xt::masked_view(phi(level, i - 1, j), mask));
        auto phi_   = xt::eval(xt::masked_view(phi(level, i, j), mask));
        auto phi_p1 = xt::eval(xt::masked_view(phi(level, i + 1, j), mask));
        auto phi_p2 = xt::eval(xt::masked_view(phi(level, i + 2, j), mask));

        auto q1 = (phi_m2 - phi_m3) * inv_dx;
        auto q2 = (phi_m1 - phi_m2) * inv_dx;
        auto q3 = (phi_ - phi_m1) * inv_dx;
        auto q4 = (phi_p1 - phi_) * inv_dx;
        auto q5 = (phi_p2 - phi_p1) * inv_dx;

        auto dphi0 = 1. / 3 * q1 - 7. / 6 * q2 + 11. / 6 * q3;
        auto dphi1 = -1. / 6 * q2 + 5. / 6 * q3 + 1. / 3 * q4;
        auto dphi2 = 1. / 3 * q3 + 5. / 6 * q4 - 1. / 6 * q5;

        auto IS0 = 13. / 12 * xt::pow(q1 - 2. * q2 + q3, 2) + 1. / 4 * xt::pow(q1 - 4 * q2 + 3 * q3, 2);
        auto IS1 = 13. / 12 * xt::pow(q2 - 2. * q3 + q4, 2) + 1. / 4 * xt::pow(q2 - q4, 2);
        auto IS2 = 13. / 12 * xt::pow(q3 - 2. * q4 + q5, 2) + 1. / 4 * xt::pow(3 * q3 - 4 * q4 + q5, 2);

        auto alpha0 = 0.1 * xt::pow((eps + IS0), -2);
        auto alpha1 = 0.6 * xt::pow((eps + IS1), -2);
        auto alpha2 = 0.3 * xt::pow((eps + IS2), -2);

        auto omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
        auto omega1 = alpha1 / (alpha0 + alpha1 + alpha2);
        auto omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

        auto tmp                    = xt::eval(xt::masked_view(dphi, mask));
        xt::masked_view(dphi, mask) = tmp + omega0 * dphi0 + omega1 * dphi1 + omega2 * dphi2;
    }

    template <class T, class Field, class Vel>
    inline auto vel_x_neg(T& dphi, const Field& phi, const Vel& vel) const
    {
        double inv_dx = 1. / dx();
        double eps    = 1e-6;
        auto mask     = vel(0, level, i, j) < 0;

        auto phi_p3 = xt::eval(xt::masked_view(phi(level, i + 3, j), mask));
        auto phi_p2 = xt::eval(xt::masked_view(phi(level, i + 2, j), mask));
        auto phi_p1 = xt::eval(xt::masked_view(phi(level, i + 1, j), mask));
        auto phi_   = xt::eval(xt::masked_view(phi(level, i, j), mask));
        auto phi_m1 = xt::eval(xt::masked_view(phi(level, i - 1, j), mask));
        auto phi_m2 = xt::eval(xt::masked_view(phi(level, i - 2, j), mask));

        auto q1 = (phi_p3 - phi_p2) * inv_dx;
        auto q2 = (phi_p2 - phi_p1) * inv_dx;
        auto q3 = (phi_p1 - phi_) * inv_dx;
        auto q4 = (phi_ - phi_m1) * inv_dx;
        auto q5 = (phi_m1 - phi_m2) * inv_dx;

        auto dphi0 = xt::eval(1. / 3 * q1 - 7. / 6 * q2 + 11. / 6 * q3);
        auto dphi1 = -1. / 6 * q2 + 5. / 6 * q3 + 1. / 3 * q4;
        auto dphi2 = 1. / 3 * q3 + 5. / 6 * q4 - 1. / 6 * q5;

        auto IS0 = 13. / 12 * xt::pow(q1 - 2. * q2 + q3, 2) + 1. / 4 * xt::pow(q1 - 4 * q2 + 3 * q3, 2);
        auto IS1 = 13. / 12 * xt::pow(q2 - 2. * q3 + q4, 2) + 1. / 4 * xt::pow(q2 - q4, 2);
        auto IS2 = 13. / 12 * xt::pow(q3 - 2. * q4 + q5, 2) + 1. / 4 * xt::pow(3 * q3 - 4 * q4 + q5, 2);

        auto alpha0 = 0.1 * xt::pow((eps + IS0), -2);
        auto alpha1 = 0.6 * xt::pow((eps + IS1), -2);
        auto alpha2 = 0.3 * xt::pow((eps + IS2), -2);

        auto omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
        auto omega1 = alpha1 / (alpha0 + alpha1 + alpha2);
        auto omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

        auto tmp                    = xt::eval(xt::masked_view(dphi, mask));
        xt::masked_view(dphi, mask) = tmp + omega0 * dphi0 + omega1 * dphi1 + omega2 * dphi2;
    }

    template <class T, class Field, class Vel>
    inline auto vel_y_pos(T& dphi, const Field& phi, const Vel& vel) const
    {
        double inv_dx = 1. / dx();
        double eps    = 1e-6;
        auto mask     = vel(1, level, i, j) >= 0;

        auto phi_m3 = xt::eval(xt::masked_view(phi(level, i, j - 3), mask));
        auto phi_m2 = xt::eval(xt::masked_view(phi(level, i, j - 2), mask));
        auto phi_m1 = xt::eval(xt::masked_view(phi(level, i, j - 1), mask));
        auto phi_   = xt::eval(xt::masked_view(phi(level, i, j), mask));
        auto phi_p1 = xt::eval(xt::masked_view(phi(level, i, j + 1), mask));
        auto phi_p2 = xt::eval(xt::masked_view(phi(level, i, j + 2), mask));

        auto q1 = (phi_m2 - phi_m3) * inv_dx;
        auto q2 = (phi_m1 - phi_m2) * inv_dx;
        auto q3 = (phi_ - phi_m1) * inv_dx;
        auto q4 = (phi_p1 - phi_) * inv_dx;
        auto q5 = (phi_p2 - phi_p1) * inv_dx;

        auto dphi0 = 1. / 3 * q1 - 7. / 6 * q2 + 11. / 6 * q3;
        auto dphi1 = -1. / 6 * q2 + 5. / 6 * q3 + 1. / 3 * q4;
        auto dphi2 = 1. / 3 * q3 + 5. / 6 * q4 - 1. / 6 * q5;

        auto IS0 = 13. / 12 * xt::pow(q1 - 2. * q2 + q3, 2) + 1. / 4 * xt::pow(q1 - 4 * q2 + 3 * q3, 2);
        auto IS1 = 13. / 12 * xt::pow(q2 - 2. * q3 + q4, 2) + 1. / 4 * xt::pow(q2 - q4, 2);
        auto IS2 = 13. / 12 * xt::pow(q3 - 2. * q4 + q5, 2) + 1. / 4 * xt::pow(3 * q3 - 4 * q4 + q5, 2);

        auto alpha0 = 0.1 * xt::pow((eps + IS0), -2);
        auto alpha1 = 0.6 * xt::pow((eps + IS1), -2);
        auto alpha2 = 0.3 * xt::pow((eps + IS2), -2);

        auto omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
        auto omega1 = alpha1 / (alpha0 + alpha1 + alpha2);
        auto omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

        auto tmp                    = xt::eval(xt::masked_view(dphi, mask));
        xt::masked_view(dphi, mask) = tmp + omega0 * dphi0 + omega1 * dphi1 + omega2 * dphi2;
    }

    template <class T, class Field, class Vel>
    inline auto vel_y_neg(T& dphi, const Field& phi, const Vel& vel) const
    {
        double inv_dx = 1. / dx();
        double eps    = 1e-6;
        auto mask     = vel(1, level, i, j) < 0;

        auto phi_p3 = xt::eval(xt::masked_view(phi(level, i, j + 3), mask));
        auto phi_p2 = xt::eval(xt::masked_view(phi(level, i, j + 2), mask));
        auto phi_p1 = xt::eval(xt::masked_view(phi(level, i, j + 1), mask));
        auto phi_   = xt::eval(xt::masked_view(phi(level, i, j), mask));
        auto phi_m1 = xt::eval(xt::masked_view(phi(level, i, j - 1), mask));
        auto phi_m2 = xt::eval(xt::masked_view(phi(level, i, j - 2), mask));

        auto q1 = (phi_p3 - phi_p2) * inv_dx;
        auto q2 = (phi_p2 - phi_p1) * inv_dx;
        auto q3 = (phi_p1 - phi_) * inv_dx;
        auto q4 = (phi_ - phi_m1) * inv_dx;
        auto q5 = (phi_m1 - phi_m2) * inv_dx;

        auto dphi0 = 1. / 3 * q1 - 7. / 6 * q2 + 11. / 6 * q3;
        auto dphi1 = -1. / 6 * q2 + 5. / 6 * q3 + 1. / 3 * q4;
        auto dphi2 = 1. / 3 * q3 + 5. / 6 * q4 - 1. / 6 * q5;

        auto IS0 = 13. / 12 * xt::pow(q1 - 2. * q2 + q3, 2) + 1. / 4 * xt::pow(q1 - 4 * q2 + 3 * q3, 2);
        auto IS1 = 13. / 12 * xt::pow(q2 - 2. * q3 + q4, 2) + 1. / 4 * xt::pow(q2 - q4, 2);
        auto IS2 = 13. / 12 * xt::pow(q3 - 2. * q4 + q5, 2) + 1. / 4 * xt::pow(3 * q3 - 4 * q4 + q5, 2);

        auto alpha0 = 0.1 * xt::pow((eps + IS0), -2);
        auto alpha1 = 0.6 * xt::pow((eps + IS1), -2);
        auto alpha2 = 0.3 * xt::pow((eps + IS2), -2);

        auto omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
        auto omega1 = alpha1 / (alpha0 + alpha1 + alpha2);
        auto omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

        auto tmp                    = xt::eval(xt::masked_view(dphi, mask));
        xt::masked_view(dphi, mask) = tmp + omega0 * dphi0 + omega1 * dphi1 + omega2 * dphi2;
    }

    template <class Field, class Vel>
    inline auto operator()(samurai::Dim<2> d, const Field& phi, const Vel& vel) const
    {
        xt::xtensor<double, 1> dphi_x = xt::zeros<double>({i.size()});
        xt::xtensor<double, 1> dphi_y = xt::zeros<double>({i.size()});
        vel_x_pos(dphi_x, phi, vel);
        vel_x_neg(dphi_x, phi, vel);
        vel_y_pos(dphi_y, phi, vel);
        vel_y_neg(dphi_y, phi, vel);
        return vel(0, level, i, j) * dphi_x + vel(1, level, i, j) * dphi_y;
    }
};

template <class... CT>
inline auto weno5(CT&&... e)
{
    return samurai::make_field_operator_function<weno5_op>(std::forward<CT>(e)...);
}

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t start_level   = 6;
    samurai::Box<double, dim> box{
        {0, 0},
        {1, 1}
    };
    AMRMesh<AMRConfig<dim>> mesh(box, start_level, start_level, start_level);

    auto field = init_field(mesh);
    auto vel   = init_velocity(mesh);

    auto field_np1 = samurai::make_field<double, 1>("sol", mesh);

    double dt           = 0.5 / (1 << start_level);
    std::size_t max_ite = 50;
    for (std::size_t ite = 0; ite < max_ite; ++ite)
    {
        update_bc(field);
        field_np1 = field - dt * weno5(field, vel);
        std::swap(field.array(), field_np1.array());
        samurai::save(fmt::format("weno_{}", ite), mesh, field);
    }
    return 0;
}
