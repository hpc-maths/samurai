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

struct init_field
{
    template <class Mesh>
    static auto call(samurai::Dim<2>, Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        auto field      = samurai::make_field<double, 1>("sol", mesh);

        samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                               [&](auto& cell)
                               {
                                   auto x        = cell.center(0);
                                   auto y        = cell.center(1);
                                   double radius = 0.15;
                                   double x0     = 0.5;
                                   double y0     = 0.75;
                                   field[cell]   = std::exp(-100 * ((x - x0) * (x - x0) + (y - y0) * (y - y0)));
                               });
        return field;
    }

    template <class Mesh>
    static auto call(samurai::Dim<3>, Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        auto field      = samurai::make_field<double, 1>("sol", mesh);

        samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                               [&](auto& cell)
                               {
                                   auto x        = cell.center(0);
                                   auto y        = cell.center(1);
                                   auto z        = cell.center(2);
                                   double radius = 0.15;
                                   double x0     = 0.5;
                                   double y0     = 0.75;
                                   double z0     = 0.5;
                                   field[cell]   = std::exp(-100. * ((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0)));
                               });
        return field;
    }
};

template <std::size_t dim, class Mesh>
auto init_velocity(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, dim>("u", mesh);
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

template <std::size_t dim, class Mesh>
auto init_phys(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto nu = samurai::make_field<double, 1>("nu", mesh);
    nu.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                               double Reynolds = 100.;

                               auto x = cell.center(0);
                               auto y = cell.center(1);

                               nu[cell] = 1. / Reynolds;
                           });

    return nu;
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

struct update_bc_for_level
{
    template <class Field>
    static void call(samurai::Dim<2>, Field& field, std::size_t level)
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

    template <class Field>
    static void call(samurai::Dim<3>, Field& field, std::size_t level)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto mesh       = field.mesh();

        std::size_t max_level = mesh[mesh_id_t::cells].max_level();

        xt::xtensor_fixed<int, xt::xshape<6, 3>> stencils{
            {-1, 0,  0 },
            {1,  0,  0 },
            {0,  -1, 0 },
            {0,  1,  0 },
            {0,  0,  -1},
            {0,  0,  1 }
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
                    auto j                = index[0];
                    auto k                = index[1];
                    field(level, i, j, k) = field(level, i - s[0], j - s[1], k - s[2]);
                });

            set_2.on(level)(
                [&](const auto& i, const auto& index)
                {
                    auto j                = index[0];
                    auto k                = index[1];
                    field(level, i, j, k) = field(level, i - 2 * s[0], j - 2 * s[1], k - 2 * s[2]);
                });

            set_3.on(level)(
                [&](const auto& i, const auto& index)
                {
                    auto j                = index[0];
                    auto k                = index[1];
                    field(level, i, j, k) = field(level, i - 3 * s[0], j - 3 * s[1], k - 3 * s[2]);
                });
        }
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

/* Convection (non-conservative) Schema WENO 5 */

template <class TInterval>
class weno5_op : public samurai::field_operator_base<TInterval>,
                 public samurai::field_expression<weno5_op<TInterval>>
{
  public:

    INIT_OPERATOR(weno5_op)

    template <class View>
    auto flux(const View& phi_i) const
    {
        double inv_dx = 1. / dx();
        double eps    = 1e-6;

        auto q1 = (xt::view(phi_i, 1) - xt::view(phi_i, 0)) * inv_dx;
        auto q2 = (xt::view(phi_i, 2) - xt::view(phi_i, 1)) * inv_dx;
        auto q3 = (xt::view(phi_i, 3) - xt::view(phi_i, 2)) * inv_dx;
        auto q4 = (xt::view(phi_i, 4) - xt::view(phi_i, 3)) * inv_dx;
        auto q5 = (xt::view(phi_i, 5) - xt::view(phi_i, 4)) * inv_dx;

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

        return xt::eval(omega0 * dphi0 + omega1 * dphi1 + omega2 * dphi2);
    }

    template <class Field, class Vel>
    auto flux_x_2d(const Field& phi, const Vel& vel) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(0, level, i, j) >= 0;

        for (int s = -3, kk = 0; s < 3; ++s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i + s, j), mask);
        }

        for (int s = 3, kk = 0; s > -3; --s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i + s, j), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }

        return flux(phi_i);
    }

    template <class Field, class Vel>
    auto flux_y_2d(const Field& phi, const Vel& vel) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(1, level, i, j) >= 0;

        for (int s = -3, kk = 0; s < 3; ++s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i, j + s), mask);
        }

        for (int s = 3, kk = 0; s > -3; --s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i, j + s), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }
        return flux(phi_i);
    }

    template <class Field, class Vel>
    auto flux_x_3d(const Field& phi, const Vel& vel) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(0, level, i, j, k) >= 0;

        for (int s = -3, kk = 0; s < 3; ++s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i + s, j, k), mask);
        }

        for (int s = 3, kk = 0; s > -3; --s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i + s, j, k), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }

        return flux(phi_i);
    }

    template <class Field, class Vel>
    auto flux_y_3d(const Field& phi, const Vel& vel) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(1, level, i, j, k) >= 0;

        for (int s = -3, kk = 0; s < 3; ++s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i, j + s, k), mask);
        }

        for (int s = 3, kk = 0; s > -3; --s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i, j + s, k), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }
        return flux(phi_i);
    }

    template <class Field, class Vel>
    auto flux_z_3d(const Field& phi, const Vel& vel) const
    {
        std::array<std::size_t, 2> shape{6, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);

        auto mask = vel(2, level, i, j, k) >= 0;

        for (int s = -3, kk = 0; s < 3; ++s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), mask) = xt::masked_view(phi(level, i, j, k + s), mask);
        }

        for (int s = 3, kk = 0; s > -3; --s, ++kk)
        {
            xt::masked_view(xt::view(phi_i, kk), !mask) = xt::masked_view(phi(level, i, j, k + s), !mask);
            xt::masked_view(xt::view(phi_i, kk), !mask) *= -1;
        }
        return flux(phi_i);
    }

    template <class Field, class Vel>
    auto operator()(samurai::Dim<2> d, const Field& phi, const Vel& vel) const
    {
        return xt::eval(vel(0, level, i, j) * flux_x_2d(phi, vel) + vel(1, level, i, j) * flux_y_2d(phi, vel));
    }

    template <class Field, class Vel>
    auto operator()(samurai::Dim<3> d, const Field& phi, const Vel& vel) const
    {
        return xt::eval(vel(0, level, i, j, k) * flux_x_3d(phi, vel) + vel(1, level, i, j, k) * flux_y_3d(phi, vel)
                        + vel(2, level, i, j, k) * flux_z_3d(phi, vel));
    }
};

template <class... CT>
inline auto weno5(CT&&... e)
{
    return samurai::make_field_operator_function<weno5_op>(std::forward<CT>(e)...);
}

/* Diffusion : Laplacien ordre 2 ou 4 */

template <class TInterval>
class laplace_explicit : public samurai::field_operator_base<TInterval>,
                         public samurai::field_expression<laplace_explicit<TInterval>>
{
  public:

    INIT_OPERATOR(laplace_explicit)

    template <class View>
    auto flux_lap(const View& phi_i, View& nu_i) const
    {
        double inv_dx = 1. / dx();

        /* Flux Ordre 2 */

        //        auto fip12 = (  xt::view(phi_i, 1)
        //        - xt::view(phi_i, 0) ) * inv_dx;

        /* Ordre 4 : fb(i+1/2) = ( -f(i+2) + 15*f(j+1) - 15*f(j) + f(j-1) )/dx
         */

        auto fip12 = (-xt::view(phi_i, 3) + 15. * xt::view(phi_i, 2) - 15. * xt::view(phi_i, 1) + xt::view(phi_i, 0)) * inv_dx;

        auto nuip12 = (xt::view(nu_i, 1) + xt::view(nu_i, 0)) * 0.5;

        return xt::eval(nuip12 * fip12);
    }

    template <class Field>
    auto Lap_x_2d(const Field& phi, const Field& nu) const
    {
        std::array<std::size_t, 2> shape{4, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);
        xt::xtensor<double, 2> nu_i  = xt::zeros<double>(shape);

        double inv_12dx = 1. / (12. * dx());

        for (int s = -2, kk = 0; s < 2; ++s, ++kk)
        {
            // xt::masked_view(xt::view(phi_i, kk), mask) =
            // xt::masked_view(phi(level, i + s, j), mask);

            // xt::masked_view(xt::view(nu_i, kk), mask) =
            // xt::masked_view(nu(level, i + s, j), mask);

            //   xt::view(phi_i, kk) = xt::view(phi(level, i + s, j));
            //   xt::view(nu_i, kk) = xt::view(nu(level, i + s, j));
            xt::view(phi_i, kk) = phi(level, i + s, j);
            xt::view(nu_i, kk)  = nu(level, i + s, j);
        }

        auto flux_D = flux_lap(phi_i, nu_i);
        flux_D *= -1.;

        for (int s = -1, kk = 0; s > 3; --s, ++kk)
        {
            //  xt::masked_view(xt::view(phi_i, kk), !mask) =
            //  xt::masked_view(phi(level, i + s, j), !mask);
            //  xt::masked_view(xt::view(nu_i, kk), !mask) =
            //  xt::masked_view(nu(level, i + s, j), !mask);
            //   xt::view(phi_i, kk) = xt::view(phi(level, i + s, j));
            //   xt::view(nu_i, kk) = xt::view(nu(level, i + s, j));
            xt::view(phi_i, kk) = phi(level, i + s, j);
            xt::view(nu_i, kk)  = nu(level, i + s, j);
        }
        flux_D += flux_lap(phi_i, nu_i);

        flux_D *= inv_12dx;

        return flux_D;
    }

    template <class Field>
    auto Lap_y_2d(const Field& phi, const Field& nu) const
    {
        std::array<std::size_t, 2> shape{4, i.size()};
        xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);
        xt::xtensor<double, 2> nu_i  = xt::zeros<double>(shape);

        double inv_12dx = 1. / (12. * dx());

        for (int s = -2, kk = 0; s < 2; ++s, ++kk)
        {
            //  xt::masked_view(xt::view(phi_i, kk), mask) =
            //  xt::masked_view(phi(level, i, j+s), mask);
            //  xt::masked_view(xt::view(nu_i, kk), mask) =
            //  xt::masked_view(nu(level, i, j+s), mask);
            xt::view(phi_i, kk) = phi(level, i, j + s);
            xt::view(nu_i, kk)  = nu(level, i, j + s);
        }

        auto flux_D = flux_lap(phi_i, nu_i);
        flux_D *= -1.;

        for (int s = -1, kk = 0; s > 3; --s, ++kk)
        {
            // xt::masked_view(xt::view(phi_i, kk), !mask) =
            // xt::masked_view(phi(level, i, j+s), !mask);
            //  xt::masked_view(xt::view(nu_i, kk), !mask) =
            //  xt::masked_view(nu(level, i, j+s), !mask);
            xt::view(phi_i, kk) = phi(level, i, j + s);
            xt::view(nu_i, kk)  = nu(level, i, j + s);
        }
        flux_D += flux_lap(phi_i, nu_i);

        flux_D *= inv_12dx;

        return flux_D;
    }

    /*
      template<class Field>
      auto Lap_x_3d(const Field& phi, const Field& nu) const
      {
          std::array<std::size_t, 2> shape{4, i.size()};
          xt::xtensor<double, 2> phi_i = xt::zeros<double>(shape);
          xt::xtensor<double, 2> nu_i = xt::zeros<double>(shape);

          double inv_12dx = 1./(12.*dx());

          for (int s = -2, kk = 0; s < 2; ++s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), mask) =
      xt::masked_view(phi(level, i + s, j, k), mask);

            xt::masked_view(xt::view(nu_i, kk), mask) =
      xt::masked_view(nu(level, i + s, j, k), mask);
          }

          double flux_D = flux_lap(phi_i, nu_i);
          flux_D *= -1.;

          for (int s = -1, kk=0; s > 3; --s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), !mask) =
      xt::masked_view(phi(level, i + s, j, k), !mask);

            xt::masked_view(xt::view(nu_i, kk), !mask) =
      xt::masked_view(nu(level, i + s, j, k), !mask);
          }
          flux_D += flux_lap(phi_i, nu_i);

          flux_D *= inv_12dx;

          return flux_D;
      }

      template<class Field>
      auto Lap_y_3d(const Field& phi, const Field& nu) const
      {
          std::array<std::size_t, 2> shape{4, i.size()};
          xt::xtensor<double, 2> nu_i = xt::zeros<double>(shape);

          double inv_12dx = 1./(12.*dx());

          for (int s = -2, kk = 0; s < 2; ++s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), mask) =
      xt::masked_view(phi(level, i, j+s, k), mask);

            xt::masked_view(xt::view(nu_i, kk), mask) =
      xt::masked_view(nu(level, i, j+s, k), mask);
          }

          double flux_D = flux_lap(phi_i, nu_i);
          flux_D *= -1.;

          for (int s = -1, kk=0; s > 3; --s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), !mask) =
      xt::masked_view(phi(level, i, j+s, k), !mask);

            xt::masked_view(xt::view(nu_i, kk), !mask) =
      xt::masked_view(nu(level, i, j+s, k), !mask);
          }
          flux_D += flux_lap(phi_i, nu_i);

          flux_D *= inv_12dx;

          return flux_D;
      }

      template<class Field>
      auto Lap_z_3d(const Field& phi, const Field& nu) const
      {
          std::array<std::size_t, 2> shape{4, i.size()};
          xt::xtensor<double, 2> nu_i = xt::zeros<double>(shape);

          double inv_12dx = 1./(12.*dx());

          for (int s = -2, kk = 0; s < 2; ++s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), mask) =
      xt::masked_view(phi(level, i, j, k+s), mask);

            xt::masked_view(xt::view(nu_i, kk), mask) =
      xt::masked_view(nu(level, i, j, k+s), mask);
          }

          double flux_D = flux_lap(phi_i, nu_i);
          flux_D *= -1.;

          for (int s = -1, kk=0; s > 3; --s, ++kk)
          {
            xt::masked_view(xt::view(phi_i, kk), !mask) =
      xt::masked_view(phi(level, i, j, k+s), !mask);

            xt::masked_view(xt::view(nu_i, kk), !mask) =
      xt::masked_view(nu(level, i, j, k+s), !mask);
          }
          flux_D += flux_lap(phi_i, nu_i);

          flux_D *= inv_12dx

          return flux_D;
      }
    */

    template <class Field>
    auto operator()(samurai::Dim<2> d, const Field& phi, const Field& nu) const
    {
        return xt::eval(Lap_x_2d(phi, nu) + Lap_y_2d(phi, nu));
    }

    /*
      template<class Field>
      auto operator()(samurai::Dim<3> d, const Field& phi, const Field& nu)
      const
      {
        return xt::eval(Lap_x_3d(phi, nu) + Lap_y_3d(phi, nu) + Lap_z_3d(phi,
      nu) );
      }
    */
};

template <class... CT>
inline auto lap_exp(CT&&... e)
{
    return samurai::make_field_operator_function<laplace_explicit>(std::forward<CT>(e)...);
}

/* Principal */

// Defining the binary function for comparison
bool comp(double a, double b)
{
    return (a < b);
}

int main()
{
    samurai::initialize();

    constexpr std::size_t dim         = 2;
    constexpr std::size_t ghost_width = 3;
    std::size_t start_level           = 7;
    std::size_t min_level             = 2;
    std::size_t max_level             = 10;

    samurai::Box<double, dim> box{
        {0, 0, 0},
        {1, 1, 1}
    };
    using Config = samurai::amr::Config<dim, ghost_width>;
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);
    using mesh_id_t = typename Config::mesh_id_t;

    auto field = init_field::call(samurai::Dim<dim>{}, mesh);

    auto update_bc = [](std::size_t level, auto& field)
    {
        update_bc_for_level::call(samurai::Dim<dim>{}, field, level);
    };

    double dx = 1. / (1 << max_level);

    double cfl = 0.5;

    std::size_t max_ite = 10;

    for (std::size_t ite = 0; ite < max_ite; ++ite)
    {
        std::cout << "iteration: " << ite << std::endl;

        std::size_t ite_adapt = 0;
        while (1)
        {
            std::cout << "\tmesh adaptation: " << ite_adapt++ << std::endl;
            samurai::update_ghost(update_bc, field);
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criteria(field, tag);
            samurai::graduation(tag, stencil_graduation::call(samurai::Dim<dim>{}));
            if (samurai::update_field(tag, field))
            {
                break;
            }
        }

        samurai::update_ghost(update_bc, field);

        auto vel = init_velocity<dim>(mesh);

        auto visc = init_phys<dim>(mesh);
        samurai::update_ghost(update_bc, visc);

        /* Calcul du pas de temps */

        double v_max    = 0.;
        double visc_max = 0.;

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](auto& cell)
                               {
                                   // v_max = std::max( vel(0, level, i, j),
                                   // v_max, comp ); v_max = std::max( vel(1,
                                   // level, i, j), v_max, comp );

                                   // visc_max = std::max( visc(level, i, j),
                                   // visc_max, comp );

                                   if (abs(vel[cell][0]) >= v_max)
                                   {
                                       v_max = abs(vel[cell][0]);
                                   }

                                   if (abs(vel[cell][1]) >= v_max)
                                   {
                                       v_max = abs(vel[cell][1]);
                                   }

                                   if (visc[cell] >= visc_max)
                                   {
                                       visc_max = visc[cell];
                                   }
                               });

        double dt = 1.e+10;
        if (cfl * dx / v_max <= 0.5 * cfl * (dx * dx) / visc_max)
        {
            dt = cfl * dx / v_max;
        }
        else
        {
            dt = 0.5 * cfl * (dx * dx) / visc_max;
        }

        /* Integration */

        auto field_np1 = samurai::make_field<double, 1>("sol", mesh);
        field_np1.fill(0.);

        /* Covection */
        field_np1 = field - dt * weno5(field, vel);

        /* Diffusion */
        field_np1 = field_np1 + dt * lap_exp(field, visc);

        std::swap(field.array(), field_np1.array());

        samurai::save(fmt::format("heat_amr_{}d_{}", dim, ite), mesh, field);
    }

    samurai::finalize();
    return 0;
}
