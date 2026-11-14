// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// NonLinear flux-based schemes: odd stencil sizes, and the fluxes at max_level.
//
// The fluxes at finer levels (--finer-level-flux, or the finer_level_flux()
// setting of a scheme) are not implemented for odd stencil sizes, except when
// the prediction stencil radius is 0 and the stencil has at most 4 cells. Such a
// scheme must still compile and run without that option, and throw when it is
// enabled.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <samurai/arguments.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

#include "test_common.hpp"

namespace samurai
{
    namespace
    {
        // Upwind flux a[d] * u(left cell of the face) on the default stencil of `size` cells.
        template <std::size_t size, class Field>
        auto make_nonlinear_upwind(const std::array<double, Field::dim>& a)
        {
            static constexpr std::size_t dim = Field::dim;
            using cfg                        = FluxConfig<SchemeType::NonLinear, size, Field, Field>;

            FluxDefinition<cfg> flux;
            static_for<0, dim>::apply(
                [&](auto integral_constant_d)
                {
                    static constexpr std::size_t d = decltype(integral_constant_d)::value;

                    double ad                  = a[d];
                    flux[d].cons_flux_function = [ad](FluxValue<cfg>& f, const StencilData<cfg>&, const StencilValues<cfg>& u)
                    {
                        // The default stencil starts at -size / 2 + 1: the left cell of the face has the index size / 2 - 1.
                        static constexpr std::size_t left = size / 2 - 1;
                        f                                 = ad * u[left];
                    };
                });
            return make_flux_based_scheme(flux);
        }

        // A periodic mesh of [0, 1]^dim, from level 2 to level 6.
        template <std::size_t dim>
        auto periodic_mesh()
        {
            using Box     = samurai::Box<double, dim>;
            using point_t = typename Box::point_t;
            point_t corner1, corner2;
            corner1.fill(0.);
            corner2.fill(1.);
            auto cfg = mesh_config<dim>().min_level(2).max_level(6).periodic(true).max_stencil_size(6);
            return mra::make_mesh(Box(corner1, corner2), cfg);
        }

        // Sets u to a Gaussian and adapts its mesh to it, which creates level jumps.
        template <class Field>
        void set_gaussian_and_adapt(Field& u)
        {
            static constexpr std::size_t dim = Field::dim;
            for_each_cell(u.mesh(),
                          [&](auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.5) * (cell.center(d) - 0.5);
                              }
                              u[cell] = std::exp(-100 * r2);
                          });
            auto adapt = make_MRAdapt(u);
            adapt(mra_config().epsilon(1e-3));
        }

        template <std::size_t dim>
        std::array<double, dim> velocity()
        {
            std::array<double, dim> a;
            for (std::size_t d = 0; d < dim; ++d)
            {
                a[d] = 1. / static_cast<double>(d + 1);
            }
            return a;
        }

        template <class Field>
        double max_difference(const Field& v, const Field& w)
        {
            double diff = 0;
            for_each_cell(v.mesh(),
                          [&](auto& cell)
                          {
                              diff = std::max(diff, std::abs(v[cell] - w[cell]));
                          });
            return diff;
        }

        // The integral of v over the mesh: zero for the divergence of a flux on a periodic mesh.
        template <class Field>
        double integral(const Field& v)
        {
            double sum = 0;
            for_each_cell(v.mesh(),
                          [&](auto& cell)
                          {
                              sum += v[cell] * std::pow(cell.length, static_cast<double>(Field::dim));
                          });
            return test::global_sum(sum);
        }

        // Restores args::finer_level_flux when a test ends.
        struct FinerLevelFluxGuard
        {
            int saved = args::finer_level_flux;

            FinerLevelFluxGuard() = default;

            FinerLevelFluxGuard(const FinerLevelFluxGuard&)            = delete;
            FinerLevelFluxGuard& operator=(const FinerLevelFluxGuard&) = delete;

            ~FinerLevelFluxGuard()
            {
                args::finer_level_flux = saved;
            }
        };

        constexpr double tol = 1e-12;
    }

    template <class T>
    class flux_based_nonlin_odd_stencil : public ::testing::Test
    {
    };

    using dims = ::testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>>;
    TYPED_TEST_SUITE(flux_based_nonlin_odd_stencil, dims, );

    // Without fluxes at finer levels, the stencils of 3 and 5 cells give the linear upwind scheme.
    TYPED_TEST(flux_based_nonlin_odd_stencil, matches_linear_upwind)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();
        VelocityVector<dim> av;
        for (std::size_t d = 0; d < dim; ++d)
        {
            av[d] = a[d];
        }

        auto linear = make_convection_upwind<Field>(av);
        auto size_2 = make_nonlinear_upwind<2, Field>(a);
        auto size_3 = make_nonlinear_upwind<3, Field>(a);
        auto size_5 = make_nonlinear_upwind<5, Field>(a);

        auto v_linear = make_scalar_field<double>("v_linear", u.mesh());
        auto v_2      = make_scalar_field<double>("v_2", u.mesh());
        auto v_3      = make_scalar_field<double>("v_3", u.mesh());
        auto v_5      = make_scalar_field<double>("v_5", u.mesh());
        v_linear      = linear(u);
        v_2           = size_2(u);
        v_3           = size_3(u);
        v_5           = size_5(u);

        EXPECT_GT(u.mesh().max_level(), u.mesh().min_level()); // the mesh has level jumps
        EXPECT_LT(max_difference(v_2, v_linear), tol);
        EXPECT_LT(max_difference(v_3, v_linear), tol);
        EXPECT_LT(max_difference(v_5, v_linear), tol);
    }

    // With fluxes at finer levels, an odd stencil size throws, whether the option is global or set on the scheme.
    TYPED_TEST(flux_based_nonlin_odd_stencil, finer_level_flux_throws)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();

        auto size_3 = make_nonlinear_upwind<3, Field>(a);
        auto size_5 = make_nonlinear_upwind<5, Field>(a);

        size_3.finer_level_flux() = 1;
        EXPECT_THROW(size_3(u), std::runtime_error);
        size_3.finer_level_flux() = 0;
        EXPECT_NO_THROW(size_3(u));

        args::finer_level_flux = 1;
        EXPECT_THROW(size_5(u), std::runtime_error);
        args::finer_level_flux = -1;
        EXPECT_THROW(size_5(u), std::runtime_error);
    }

    // An even stencil size keeps its fluxes at finer levels: the stencils of 2 and 4 cells agree,
    // and the scheme is conservative.
    TYPED_TEST(flux_based_nonlin_odd_stencil, even_stencil_with_finer_level_flux)
    {
        static constexpr std::size_t dim = TypeParam::value;
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 1;

        auto mesh = periodic_mesh<dim>();
        auto u    = make_scalar_field<double>("u", mesh);
        set_gaussian_and_adapt(u);
        using Field = decltype(u);
        auto a      = velocity<dim>();

        auto size_2 = make_nonlinear_upwind<2, Field>(a);
        auto size_4 = make_nonlinear_upwind<4, Field>(a);

        auto v_2 = make_scalar_field<double>("v_2", u.mesh());
        auto v_4 = make_scalar_field<double>("v_4", u.mesh());
        EXPECT_NO_THROW(v_2 = size_2(u));
        EXPECT_NO_THROW(v_4 = size_4(u));

        EXPECT_LT(max_difference(v_2, v_4), tol);
        EXPECT_LT(std::abs(integral(v_2)), tol);
    }

    // Fluxes at max_level on a 3-level mesh (levels 2, 3, 4): an interface at level l sums the fluxes
    // of the 2^((dim-1)*(4-l)) transverse children of the interface, with gaps of 1 and 2 levels.
    namespace
    {
        constexpr std::size_t flf_min_level = 2;
        constexpr std::size_t flf_max_level = 4;

        // Level of the cells around x in a nested mesh of [0, 1]^dim: level 4 in [3/8, 5/8)^dim,
        // level 3 around it up to [1/4, 3/4)^dim, level 2 elsewhere.
        template <std::size_t dim>
        std::size_t nested_level(const std::array<double, dim>& x)
        {
            auto inside = [&](double a, double b)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (x[d] < a || x[d] >= b)
                    {
                        return false;
                    }
                }
                return true;
            };
            return inside(0.375, 0.625) ? 4 : (inside(0.25, 0.75) ? 3 : 2);
        }

        int process_rank()
        {
#ifdef SAMURAI_WITH_MPI
            return boost::mpi::communicator().rank();
#else
            return 0;
#endif
        }

        // Non-periodic mesh of [0, 1]^dim with the levels of nested_level(). The cells of level 2
        // are dealt out to the processes in slabs, then each process refines its own.
        template <std::size_t dim>
        auto three_level_mesh()
        {
            using Box     = samurai::Box<double, dim>;
            using point_t = typename Box::point_t;
            point_t corner1, corner2;
            corner1.fill(0.);
            corner2.fill(1.);
            auto cfg  = mesh_config<dim>().min_level(flf_min_level).max_level(flf_max_level).max_stencil_size(4);
            auto base = mra::make_mesh(Box(corner1, corner2), cfg); // gives the domain and the configuration

            const int n_coarse = 1 << flf_min_level;
            int n_coarse_cells = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                n_coarse_cells *= n_coarse;
            }

            using Mesh = decltype(base);
            typename Mesh::cl_type cl;
            for (int coarse = 0; coarse < n_coarse_cells; ++coarse)
            {
                if (coarse * test::nb_processes() / n_coarse_cells != process_rank())
                {
                    continue;
                }
                for (std::size_t level = flf_min_level; level <= flf_max_level; ++level)
                {
                    const int r    = 1 << (level - flf_min_level); // children per direction
                    int n_children = 1;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        n_children *= r;
                    }
                    for (int child = 0; child < n_children; ++child)
                    {
                        std::array<int, dim> i;
                        std::array<double, dim> center;
                        for (std::size_t d = 0, c = static_cast<std::size_t>(coarse), k = static_cast<std::size_t>(child); d < dim; ++d)
                        {
                            const auto nc = static_cast<std::size_t>(n_coarse);
                            const auto nr = static_cast<std::size_t>(r);
                            i[d]          = static_cast<int>((c % nc) * nr + k % nr);
                            center[d]     = (i[d] + 0.5) / (n_coarse * r);
                            c /= nc;
                            k /= nr;
                        }
                        if (nested_level<dim>(center) == level)
                        {
                            xt::xtensor_fixed<int, xt::xshape<dim - 1>> yz;
                            for (std::size_t d = 1; d < dim; ++d)
                            {
                                yz[d - 1] = i[d];
                            }
                            cl[level][yz].add_interval({i[0], i[0] + 1});
                        }
                    }
                }
            }
            return Mesh(cl, base);
        }

        template <class Mesh>
        void expect_all_levels(const Mesh& mesh)
        {
            for (std::size_t level = flf_min_level; level <= flf_max_level; ++level)
            {
                EXPECT_GT(test::global_sum(mesh[Mesh::mesh_id_t::cells][level].nb_cells()), 0) << "no cell at level " << level;
            }
        }

        // Sets a field in every cell and ghost of its mesh, at every level, so that the scheme
        // uses these values as they are, with no exchange between processes. A boundary
        // condition would spoil the corner ghosts.
        template <class Field, class Func>
        void set_exactly(Field& field, Func&& set_cell)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;
            for_each_cell(field.mesh()[mesh_id_t::reference], std::forward<Func>(set_cell));
            field.ghosts_updated() = true;
        }

        // u = c . x, reproduced exactly by the prediction.
        template <std::size_t dim>
        double linear(const std::array<double, dim>& c, const auto& x)
        {
            double v = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                v += c[d] * x[d];
            }
            return v;
        }
    }

    template <class T>
    class flux_based_nonlin_max_level_flux : public ::testing::Test
    {
    };

    using flf_dims = ::testing::Types<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;
    TYPED_TEST_SUITE(flux_based_nonlin_max_level_flux, flf_dims, );

    // Every max_level face of every interface inside the domain gets exactly one flux evaluation
    // on each process that holds a cell next to it.
    // The field holds the cell center, so the left value of a flux gives its fine cell.
    template <std::size_t stencil_size, std::size_t dim>
    void check_each_fine_face_once()
    {
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = three_level_mesh<dim>();
        expect_all_levels(mesh);

        auto x = make_vector_field<double, dim>("x", mesh);
        set_exactly(x,
                    [&](auto& cell)
                    {
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            x[cell][d] = cell.center(d);
                        }
                    });

        using Field = decltype(x);
        using cfg   = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

        // count[d][fine cell on the left of the face]
        const int n_fine = 1 << flf_max_level;
        const double h   = 1. / n_fine;
        std::array<std::map<std::array<int, dim>, int>, dim> count;
        std::mutex count_mutex; // with OpenMP, the scheme calls the flux function from several threads

        FluxDefinition<cfg> flux;
        static_for<0, dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;
                flux[d].cons_flux_function     = [&](FluxValue<cfg>& f, const StencilData<cfg>&, const StencilValues<cfg>& u)
                {
                    f = u[stencil_size / 2 - 1];
                    std::array<int, dim> index;
                    for (std::size_t k = 0; k < dim; ++k)
                    {
                        double i = f[k] / h - 0.5;
                        index[k] = static_cast<int>(std::lround(i));
                        if (std::abs(i - index[k]) > 1e-8 || index[k] < 0 || index[k] >= n_fine)
                        {
                            return; // a face that is not at max_level, or on the boundary
                        }
                    }
                    if (index[d] < n_fine - 1)
                    {
                        std::lock_guard<std::mutex> lock(count_mutex);
                        ++count[d][index];
                    }
                };
            });
        auto scheme = make_flux_based_scheme(flux);
        scheme.enable_max_level_flux(true);

        auto v = make_vector_field<double, dim>("v", mesh);
        v      = scheme(x);

        // Expected: the max_level faces on both sides of every local cell, inside the domain.
        std::array<std::set<std::array<int, dim>>, dim> expected;
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          const int r    = 1 << (flf_max_level - cell.level); // children per direction
                          int n_subfaces = 1;
                          for (std::size_t k = 1; k < dim; ++k)
                          {
                              n_subfaces *= r;
                          }
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              for (int m = 0; m < n_subfaces; ++m)
                              {
                                  std::array<int, dim> index;
                                  for (std::size_t k = 0, rest = static_cast<std::size_t>(m); k < dim; ++k)
                                  {
                                      index[k] = static_cast<int>(cell.indices[k]) * r;
                                      if (k != d)
                                      {
                                          index[k] += static_cast<int>(rest % static_cast<std::size_t>(r));
                                          rest /= static_cast<std::size_t>(r);
                                      }
                                  }
                                  // fine cell on the left of the left face, then of the right face
                                  for (int left : {index[d] - 1, index[d] + r - 1})
                                  {
                                      index[d] = left;
                                      if (left >= 0 && left < n_fine - 1)
                                      {
                                          expected[d].insert(index);
                                      }
                                  }
                              }
                          }
                      });

        for (std::size_t d = 0; d < dim; ++d)
        {
            std::size_t n_missing  = 0;
            std::size_t n_repeated = 0;
            for (const auto& index : expected[d])
            {
                auto it = count[d].find(index);
                n_missing += (it == count[d].end());
                n_repeated += (it != count[d].end() && it->second != 1);
            }
            EXPECT_EQ(n_missing, 0) << "direction " << d;
            EXPECT_EQ(n_repeated, 0) << "direction " << d;
            EXPECT_EQ(count[d].size(), expected[d].size()) << "direction " << d; // nothing unexpected
        }
    }

    TYPED_TEST(flux_based_nonlin_max_level_flux, each_fine_face_once)
    {
        static constexpr std::size_t dim = TypeParam::value;
        check_each_fine_face_once<2, dim>();
        check_each_fine_face_once<4, dim>();
    }

    // With u = c . x and the upwind flux a[d] * u(left), the face fluxes at max_level are exact, so
    // the scheme gives sum_d a[d] * c[d] in every cell that does not touch the boundary.
    template <std::size_t stencil_size, std::size_t dim>
    void check_linear_field_divergence()
    {
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        auto mesh = three_level_mesh<dim>();
        expect_all_levels(mesh);

        std::array<double, dim> c;
        for (std::size_t d = 0; d < dim; ++d)
        {
            c[d] = static_cast<double>(1 << (2 * d)); // 1, 4, 16
        }
        auto u = make_scalar_field<double>("u", mesh);
        set_exactly(u,
                    [&](auto& cell)
                    {
                        u[cell] = linear<dim>(c, cell.center());
                    });

        using Field = decltype(u);
        auto a      = velocity<dim>();
        auto scheme = make_nonlinear_upwind<stencil_size, Field>(a);
        scheme.enable_max_level_flux(true);

        auto v = make_scalar_field<double>("v", mesh);
        v      = scheme(u);

        double expected = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            expected += a[d] * c[d];
        }
        double error = 0;
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              if (cell.corner(d) < tol || cell.corner(d) + cell.length > 1 - tol)
                              {
                                  return;
                              }
                          }
                          error = std::max(error, std::abs(v[cell] - expected));
                      });
        EXPECT_LT(test::global_max(error), 1e-10 * expected);
    }

    TYPED_TEST(flux_based_nonlin_max_level_flux, linear_field_divergence)
    {
        static constexpr std::size_t dim = TypeParam::value;
        check_linear_field_divergence<2, dim>();
        check_linear_field_divergence<4, dim>();
    }
}

namespace samurai
{
    // The finer-level fluxes predict their stencil values with 1D matrices, one along the flux
    // direction and one across it. Their tensor product must give the prediction maps of
    // portion(), and must read the same coarse cells: no fewer (missing weights), no more (a
    // cell that may not exist, e.g. near a subdomain boundary with MPI).
    template <std::size_t radius, std::size_t dim>
    void check_prediction_matrices(std::size_t delta_l, std::size_t stencil_size, std::size_t d)
    {
        using key_t = std::array<int, dim>;

        const int n_children = 1 << delta_l;
        const auto along     = detail::prediction_matrix_1d<radius>(delta_l, -static_cast<int>(stencil_size / 2), stencil_size);
        const auto across    = detail::prediction_matrix_1d<radius>(delta_l, 0, static_cast<std::size_t>(n_children));

        // Weights read by every fine cell of the interface, per (fine cell, coarse cell).
        std::size_t n_fine = stencil_size;
        for (std::size_t k = 1; k < dim; ++k)
        {
            n_fine *= static_cast<std::size_t>(n_children);
        }
        std::map<key_t, bool> portion_cells;
        double max_error = 0;
        for (std::size_t f = 0; f < n_fine; ++f)
        {
            // fine cell: row[d] along the flux direction, a child index in the others
            std::array<std::size_t, dim> row;
            key_t fine;
            key_t coarse_shift;
            for (std::size_t k = 0, rest = f; k < dim; ++k)
            {
                const std::size_t n = k == d ? stencil_size : static_cast<std::size_t>(n_children);
                row[k]              = rest % n;
                rest /= n;
                const int t     = (k == d ? -static_cast<int>(stencil_size / 2) : 0) + static_cast<int>(row[k]);
                fine[k]         = t & (n_children - 1);
                coarse_shift[k] = t >> delta_l;
            }
            const auto& map = std::apply(
                [&](auto... i) -> const auto&
                {
                    return prediction<radius>(delta_l, i...);
                },
                fine);
            for (const auto& [offset, weight] : map.coeff)
            {
                key_t cell;
                double tensor_weight = 1;
                bool in_box          = true;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    cell[k]       = offset[k] + coarse_shift[k];
                    const auto& m = k == d ? along : across;
                    const int col = cell[k] - m.lo;
                    in_box        = in_box && col >= 0 && col < static_cast<int>(m.n_cols);
                    tensor_weight *= in_box ? m(row[k], static_cast<std::size_t>(col)) : 0.;
                }
                ASSERT_TRUE(in_box) << "portion() reads a cell out of the box";
                portion_cells[cell] = true;
                max_error           = std::max(max_error, std::abs(tensor_weight - weight));
            }
        }
        EXPECT_LT(max_error, 1e-14);

        // Every cell of the box is read by portion().
        std::size_t n_box = 1;
        for (std::size_t k = 0; k < dim; ++k)
        {
            n_box *= k == d ? along.n_cols : across.n_cols;
        }
        EXPECT_EQ(portion_cells.size(), n_box);
    }

    TEST(flux_based_nonlin_prediction_matrices, tensor_product_of_portion)
    {
        for (std::size_t stencil_size : std::array<std::size_t, 3>{2, 4, 6})
        {
            for (std::size_t delta_l = 1; delta_l <= 3; ++delta_l)
            {
                check_prediction_matrices<0, 2>(delta_l, stencil_size, 0);
                check_prediction_matrices<1, 2>(delta_l, stencil_size, 0);
                check_prediction_matrices<1, 2>(delta_l, stencil_size, 1);
                check_prediction_matrices<2, 2>(delta_l, stencil_size, 1);
                check_prediction_matrices<1, 3>(delta_l, stencil_size, 0);
                check_prediction_matrices<1, 3>(delta_l, stencil_size, 2);
                check_prediction_matrices<2, 3>(delta_l, stencil_size, 1);
            }
        }
    }

    // Every fine stencil value the flux function receives, compared to portion() cell by cell, on
    // a mesh refined at the domain boundary.
    namespace
    {
        // A field that varies sharply at the corner 0 of [0, 1]^dim, so the mesh is refined there.
        template <std::size_t dim>
        double corner_bump(const auto& x, std::size_t comp)
        {
            double r2 = 0;
            for (std::size_t k = 0; k < dim; ++k)
            {
                r2 += x[k] * x[k];
            }
            return 2. + static_cast<double>(comp) + std::exp(-40 * r2) * (1. + static_cast<double>(comp) * x[0]);
        }

        // Prediction of the child `child` of the coarse cell `coarse`, delta_l levels finer.
        template <class Field, std::size_t... K>
        double predicted_value(const Field& u,
                               [[maybe_unused]] std::size_t comp,
                               std::size_t level,
                               std::size_t delta_l,
                               const std::array<int, Field::dim>& coarse,
                               const std::array<int, Field::dim>& child,
                               std::index_sequence<K...>)
        {
            using interval_t = typename Field::interval_t;
            auto i           = std::make_tuple(interval_t{coarse[0], coarse[0] + 1}, coarse[K + 1]...);
            auto ii          = std::make_tuple(child[0], child[K + 1]...);
            if constexpr (Field::is_scalar)
            {
                return portion(u, level, delta_l, i, ii)(0);
            }
            else
            {
                return portion(u, comp, level, delta_l, i, ii)(0);
            }
        }
    }

    template <std::size_t stencil_size, std::size_t n_comp, std::size_t dim, int radius = 1>
    void check_predicted_stencil_values(bool periodic, std::size_t max_level)
    {
        FinerLevelFluxGuard guard;
        args::finer_level_flux = 0;

        using Box     = samurai::Box<double, dim>;
        using point_t = typename Box::point_t;
        point_t corner1, corner2;
        corner1.fill(0.);
        corner2.fill(1.);
        auto cfg  = mesh_config<dim, radius>().min_level(2).max_level(max_level).periodic(periodic).max_stencil_size(6);
        auto mesh = mra::make_mesh(Box(corner1, corner2), cfg);

        auto make_field = [&]()
        {
            if constexpr (n_comp == 1)
            {
                return make_scalar_field<double>("u", mesh);
            }
            else
            {
                return make_vector_field<double, n_comp>("u", mesh);
            }
        };
        auto u      = make_field();
        using Field = decltype(u);
        auto init   = [&]()
        {
            for_each_cell(u.mesh(),
                          [&](auto& cell)
                          {
                              for (std::size_t c = 0; c < n_comp; ++c)
                              {
                                  field_value(u, cell, c) = corner_bump<dim>(cell.center(), c);
                              }
                          });
        };
        init();
        if (!periodic)
        {
            if constexpr (n_comp == 1)
            {
                make_bc<Dirichlet<1>>(u, 3.);
            }
            else
            {
                make_bc<Dirichlet<1>>(u, 3., 4.);
            }
        }
        auto adapt = make_MRAdapt(u);
        adapt(mra_config().epsilon(1e-4));
        init();
        u.ghosts_updated() = false;

        // The mesh is refined at the boundary x = 0.
        using mesh_id_t           = typename decltype(mesh)::mesh_id_t;
        const std::size_t finest  = mesh[mesh_id_t::cells].max_level();
        std::size_t n_finest_left = 0;
        for_each_cell(mesh[mesh_id_t::cells][finest],
                      [&](auto& cell)
                      {
                          n_finest_left += cell.indices[0] == 0;
                      });
        EXPECT_GT(test::global_max(finest), test::global_min(mesh[mesh_id_t::cells].min_level())) << "no level jump";
        EXPECT_GT(test::global_sum(n_finest_left), 0) << "not refined at the boundary";

        using cfg_t = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

        // The fine fluxes of an interface come one after the other, the transverse children
        // counted like an odometer, the lowest direction first. One per thread: with OpenMP a
        // thread handles whole interfaces.
        struct Interface
        {
            std::size_t d     = 0;
            std::size_t level = 0;
            std::array<int, dim> right{};
            std::size_t fine_flux = 0;
        };

        thread_local Interface current;

        std::size_t n_checked = 0;
        std::size_t n_wrong   = 0;
        double max_error      = 0;
        std::mutex result_mutex; // with OpenMP, the scheme calls the flux function from several threads

        FluxDefinition<cfg_t> flux;
        static_for<0, dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;
                flux[d].cons_flux_function = [&](FluxValue<cfg_t>& f, const StencilData<cfg_t>& data, const StencilValues<cfg_t>& values)
                {
                    if constexpr (n_comp == 1)
                    {
                        f = 0;
                    }
                    else
                    {
                        f.fill(0);
                    }

                    const auto& right_cell    = data.cells[stencil_size / 2];
                    const std::size_t level   = right_cell.level;
                    const std::size_t delta_l = u.mesh().max_level() - level;
                    const int n_children      = 1 << delta_l;
                    std::size_t n_fine_fluxes = 1;
                    std::array<int, dim> right;
                    for (std::size_t k = 0; k < dim; ++k)
                    {
                        right[k] = right_cell.indices[k];
                        n_fine_fluxes *= k == d ? 1 : static_cast<std::size_t>(n_children);
                    }
                    if (current.d != d || current.level != level || current.right != right || current.fine_flux == n_fine_fluxes)
                    {
                        current = {d, level, right, 0};
                    }
                    const std::size_t fine_flux = current.fine_flux++;

                    std::size_t flux_n_wrong = 0;
                    double flux_max_error    = 0;

                    for (std::size_t s = 0; s < stencil_size; ++s)
                    {
                        std::array<int, dim> coarse = right;
                        std::array<int, dim> child;
                        for (std::size_t k = 0, rest = fine_flux; k < dim; ++k)
                        {
                            if (k == d)
                            {
                                const int t = static_cast<int>(s) - static_cast<int>(stencil_size / 2);
                                coarse[k] += t >> delta_l;
                                child[k] = t & (n_children - 1);
                            }
                            else
                            {
                                child[k] = static_cast<int>(rest % static_cast<std::size_t>(n_children));
                                rest /= static_cast<std::size_t>(n_children);
                            }
                        }
                        for (std::size_t c = 0; c < n_comp; ++c)
                        {
                            const double expected = predicted_value(u, c, level, delta_l, coarse, child, std::make_index_sequence<dim - 1>{});
                            double value = 0;
                            if constexpr (n_comp == 1)
                            {
                                value = values[s];
                            }
                            else
                            {
                                value = values[s](c);
                            }
                            const double error = std::abs(value - expected) / std::abs(expected);
                            flux_max_error     = std::max(flux_max_error, error);
                            flux_n_wrong += error > 1e-13;
                        }
                    }

                    std::lock_guard<std::mutex> lock(result_mutex);
                    n_checked += stencil_size * n_comp;
                    n_wrong += flux_n_wrong;
                    max_error = std::max(max_error, flux_max_error);
                };
            });
        auto scheme = make_flux_based_scheme(flux);
        scheme.include_boundary_fluxes(false); // the boundary fluxes are not computed at finer levels
        scheme.finer_level_flux() = -1;

        current = Interface{};
        auto v  = make_field();
        v       = scheme(u);

        EXPECT_GT(test::global_sum(n_checked), 0);
        EXPECT_EQ(test::global_sum(n_wrong), 0) << "max relative error " << max_error;
    }

    template <class T>
    class flux_based_nonlin_predicted_stencil : public ::testing::Test
    {
    };

    using predicted_stencil_dims = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;
    TYPED_TEST_SUITE(flux_based_nonlin_predicted_stencil, predicted_stencil_dims, );

    TYPED_TEST(flux_based_nonlin_predicted_stencil, matches_portion)
    {
        static constexpr std::size_t dim = TypeParam::value;
        const std::size_t max_level      = dim == 3 ? 4 : 6;
        for (bool periodic : {false, true})
        {
            SCOPED_TRACE(periodic ? "periodic" : "Dirichlet");
            check_predicted_stencil_values<2, 1, dim>(periodic, max_level);
            check_predicted_stencil_values<4, 1, dim>(periodic, max_level);
            check_predicted_stencil_values<6, 1, dim>(periodic, max_level);
            check_predicted_stencil_values<2, 2, dim>(periodic, max_level);
            check_predicted_stencil_values<4, 2, dim>(periodic, max_level);
            check_predicted_stencil_values<6, 2, dim>(periodic, max_level);
        }
        // A prediction stencil radius of 0 (with 4 cells or less, the coarse values are copied instead).
        check_predicted_stencil_values<6, 1, dim, 0>(false, max_level);
    }
}
