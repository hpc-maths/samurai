// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// StdArrayWrapper: the field values a flux function reads, and the default
// flux container. The tests are written outside the samurai namespace, the way
// user code is: the functions on arrays must be found without qualification.

#include <array>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/storage/std/algebraic_array.hpp>

namespace
{
    using array3   = samurai::StdArrayWrapper<double, 3>;
    using xfixed3  = xt::xtensor_fixed<double, xt::xshape<3>>;
    using ixfixed3 = xt::xtensor_fixed<int, xt::xshape<3>>;

    template <class Expression>
    constexpr bool is_array3 = std::is_same_v<Expression, array3>;

    void expect_values(const array3& a, double a0, double a1, double a2)
    {
        EXPECT_DOUBLE_EQ(a[0], a0);
        EXPECT_DOUBLE_EQ(a[1], a1);
        EXPECT_DOUBLE_EQ(a[2], a2);
    }

    double first_value(const xfixed3& x)
    {
        return x(0);
    }

    auto make_mesh()
    {
        const samurai::Box<double, 1> box{{0}, {1}};
        return samurai::mra::make_mesh(box, samurai::mesh_config<1>().min_level(3).max_level(3));
    }
}

//--------------//
// Construction //
//--------------//

TEST(algebraic_array, construction)
{
    expect_values(array3(), 0, 0, 0);
    expect_values(array3(std::array<double, 3>{1, 2, 3}), 1, 2, 3);
    expect_values(array3{1, 2, 3}, 1, 2, 3); // a list of int
    expect_values(array3(4.), 4, 4, 4);

    const array3 a{1., 2., 3.};
    const array3 braces{a}; // a copy, not a list of one array
    expect_values(braces, 1, 2, 3);

    array3 b;
    b = 5.;
    expect_values(b, 5, 5, 5);
    b.fill(6.);
    expect_values(b, 6, 6, 6);
}

TEST(algebraic_array, element_access)
{
    array3 a{1., 2., 3.};
    a(0) = 10.;
    a[1] = 20.;
    expect_values(a, 10, 20, 3);
    EXPECT_EQ(a.size(), 3u);
    EXPECT_EQ(*a.data(), 10.);
    EXPECT_EQ(a.array()[2], 3.);

    double total = 0;
    for (const double value : a)
    {
        total += value;
    }
    EXPECT_DOUBLE_EQ(total, 33.);

    for (double& value : a)
    {
        value = 1.;
    }
    expect_values(a, 1, 1, 1);
}

//------------//
// Arithmetic //
//------------//

TEST(algebraic_array, array_with_array)
{
    const array3 a{1., 2., 3.};
    const array3 b{4., 8., 6.};

    expect_values(a + b, 5, 10, 9);
    expect_values(a - b, -3, -6, -3);
    expect_values(b - a, 3, 6, 3);
    expect_values(a * b, 4, 16, 18);
    expect_values(a / b, 0.25, 0.25, 0.5);
    expect_values(b / a, 4, 4, 2);
    expect_values(-a, -1, -2, -3);
}

TEST(algebraic_array, array_with_scalar)
{
    const array3 a{1., 2., 4.};

    expect_values(a + 1., 2, 3, 5);
    expect_values(1. + a, 2, 3, 5);
    expect_values(a - 1., 0, 1, 3);
    expect_values(1. - a, 0, -1, -3);
    expect_values(a * 2., 2, 4, 8);
    expect_values(2. * a, 2, 4, 8);
    expect_values(a / 2., 0.5, 1, 2);
    expect_values(2. / a, 2, 1, 0.5);

    // An integer scalar does not truncate the values.
    expect_values(a / 2, 0.5, 1, 2);
    expect_values(3 * a, 3, 6, 12);
}

TEST(algebraic_array, compound_assignment)
{
    const array3 b{4., 8., 6.};
    array3 a{1., 2., 3.};

    a += b;
    expect_values(a, 5, 10, 9);
    a -= b;
    expect_values(a, 1, 2, 3);
    a *= b;
    expect_values(a, 4, 16, 18);
    a /= b;
    expect_values(a, 1, 2, 3);

    a += 1.;
    expect_values(a, 2, 3, 4);
    a -= 1.;
    expect_values(a, 1, 2, 3);
    a *= 2.;
    expect_values(a, 2, 4, 6);
    a /= 2.;
    expect_values(a, 1, 2, 3);
}

// A flux function chains operations on temporaries. Each one must return a new
// array: an operator handing back a reference to one of its temporary operands
// dangles as soon as the result is bound to 'const auto&'.
TEST(algebraic_array, every_operation_returns_a_value)
{
    array3 a{1., 2., 3.};
    array3 b{4., 8., 6.};

    static_assert(is_array3<decltype(a + b)>);
    static_assert(is_array3<decltype(std::move(a) + b)>);
    static_assert(is_array3<decltype(a + std::move(b))>);
    static_assert(is_array3<decltype(std::move(a) + std::move(b))>);
    static_assert(is_array3<decltype(std::move(a) - b)>);
    static_assert(is_array3<decltype(std::move(a) * b)>);
    static_assert(is_array3<decltype(std::move(a) / b)>);
    static_assert(is_array3<decltype(2. * std::move(a))>);
    static_assert(is_array3<decltype(std::move(a) * 2.)>);
    static_assert(is_array3<decltype(std::move(a) / 2.)>);
    static_assert(is_array3<decltype(-std::move(a))>);
    static_assert(is_array3<decltype(pow(std::move(a), 2))>);
    static_assert(is_array3<decltype(abs(std::move(a)))>);

    const auto& flux = 0.5 * (a + b) - 2. * (b - a) / 4.;
    const array3 noise{-1., -1., -1.}; // reuses the stack of the temporaries, if any dangles
    expect_values(flux, 1, 2, 3);
    expect_values(noise, -1, -1, -1);
}

//-----------//
// Functions //
//-----------//

TEST(algebraic_array, element_wise_functions)
{
    const array3 a{-1., 4., -9.};
    const array3 b{0., 5., -10.};

    expect_values(abs(a), 1, 4, 9);
    expect_values(sqrt(abs(a)), 1, 2, 3);
    expect_values(pow(a, 2), 1, 16, 81);
    expect_values(minimum(a, b), -1, 4, -10);
    expect_values(maximum(a, b), 0, 5, -9);

    // A scalar on either side stands for a constant array.
    expect_values(minimum(a, 0.), -1, 0, -9);
    expect_values(maximum(0., a), 0, 4, 0);
}

TEST(algebraic_array, reductions)
{
    const array3 a{3., -4., 12.};

    EXPECT_DOUBLE_EQ(sum(a), 11.);
    EXPECT_DOUBLE_EQ(amin(a), -4.);
    EXPECT_DOUBLE_EQ(amax(a), 12.);
    EXPECT_DOUBLE_EQ(norm_l2(a), 13.);
}

TEST(algebraic_array, comparison_and_output)
{
    const array3 a{1., 2., 3.};
    const array3 b{1., 2., 4.};

    EXPECT_TRUE(a == a);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    std::ostringstream out;
    out << a;
    EXPECT_EQ(out.str(), "{1, 2, 3}");
}

// The functions on arrays are friends, found through their argument only: the
// samurai namespace still reaches the functions on scalars.
namespace samurai
{
    TEST(algebraic_array, scalar_functions_stay_reachable_in_samurai)
    {
        EXPECT_DOUBLE_EQ(sqrt(16.), 4.);
        EXPECT_DOUBLE_EQ(abs(-2.), 2.);
        EXPECT_DOUBLE_EQ(pow(3., 2), 9.);
    }
}

//---------//
// xtensor //
//---------//

TEST(algebraic_array, from_xtensor_expression)
{
    const xfixed3 x{1., 2., 3.};
    const ixfixed3 n{4, 5, 6};
    const xt::xtensor<double, 2> matrix{
        {1., 2., 3.},
        {4., 5., 6.}
    };

    expect_values(array3(x), 1, 2, 3);                   // container
    expect_values(array3(x + 2. * x), 3, 6, 9);          // function
    expect_values(array3(xt::view(matrix, 1)), 4, 5, 6); // view
    expect_values(array3(n), 4, 5, 6);                   // another value type

    array3 a;
    a = x;
    expect_values(a, 1, 2, 3);
    a = x * x;
    expect_values(a, 1, 4, 9);
    a = xt::view(matrix, 1);
    expect_values(a, 4, 5, 6);
}

TEST(algebraic_array, to_xtensor)
{
    const array3 a{3., -4., 12.};

    const xfixed3 implicit = a;
    EXPECT_EQ(implicit, (xfixed3{3., -4., 12.}));

    xfixed3 assigned;
    assigned = a;
    EXPECT_EQ(assigned, (xfixed3{3., -4., 12.}));

    // A function with an xtensor parameter.
    EXPECT_DOUBLE_EQ(first_value(a), 3.);

    // Any xtensor function, on an explicit copy.
    EXPECT_DOUBLE_EQ(xt::norm_l2(a.to_xtensor())(), 13.);
    EXPECT_DOUBLE_EQ(xt::view(a.to_xtensor(), xt::range(1, 3))(1), 12.);

    EXPECT_TRUE(a == implicit);
    EXPECT_TRUE(implicit == a);
}

// xtensor takes any type it does not know for a scalar. An array next to an
// xtensor expression must be combined component by component, into an array.
TEST(algebraic_array, mixed_with_xtensor_expression)
{
    const array3 a{1., 2., 4.};
    const xfixed3 x{8., 4., 2.};
    const ixfixed3 n{1, 2, 3};

    static_assert(is_array3<decltype(a + x)>);
    static_assert(is_array3<decltype(x + a)>);
    static_assert(is_array3<decltype(a - (x + x))>);
    static_assert(is_array3<decltype((x + x) - a)>);
    static_assert(is_array3<decltype(a * xt::view(x, xt::all()))>);
    static_assert(is_array3<decltype(x / array3())>);

    expect_values(a + x, 9, 6, 6);
    expect_values(x + a, 9, 6, 6);
    expect_values(a - x, -7, -2, 2);
    expect_values(x - a, 7, 2, -2);
    expect_values(a * x, 8, 8, 8);
    expect_values(x * a, 8, 8, 8);
    expect_values(a / x, 0.125, 0.5, 2);
    expect_values(x / a, 8, 2, 0.5);

    // A lazy xtensor function, an array temporary, another value type.
    expect_values((x + x) - a, 15, 6, 0);
    expect_values(0.5 * (x + x - 2. * (a - a)), 8, 4, 2);
    expect_values(n * a, 1, 4, 12);

    array3 b{1., 2., 4.};
    b += x;
    expect_values(b, 9, 6, 6);
    b -= x + x;
    expect_values(b, -7, -2, 2);
    b *= x;
    expect_values(b, -56, -8, 4);
    b /= x;
    expect_values(b, -7, -2, 2);
}

TEST(algebraic_array, compound_assignment_to_xtensor_expression)
{
    const array3 a{1., 2., 4.};

    xfixed3 x{8., 4., 2.};
    x += a;
    EXPECT_EQ(x, (xfixed3{9., 6., 6.}));
    x -= a;
    EXPECT_EQ(x, (xfixed3{8., 4., 2.}));
    x *= a;
    EXPECT_EQ(x, (xfixed3{8., 8., 8.}));
    x /= a;
    EXPECT_EQ(x, (xfixed3{8., 4., 2.}));

    // A temporary view: one column of a matrix.
    xt::xtensor_fixed<double, xt::xshape<3, 2>> matrix;
    matrix.fill(1.);
    xt::col(matrix, 1) -= a;
    EXPECT_EQ(xt::col(matrix, 0), (xfixed3{1., 1., 1.}));
    EXPECT_EQ(xt::col(matrix, 1), (xfixed3{0., -1., -3.}));
}

//--------//
// Fields //
//--------//

TEST(algebraic_array, field_value_types)
{
    const auto mesh = make_mesh();
    auto u          = samurai::make_vector_field<double, 3>("u", mesh);
    auto s          = samurai::make_scalar_field<double>("s", mesh);
    auto one        = samurai::make_vector_field<double, 1>("one", mesh);

    static_assert(std::is_same_v<decltype(u)::local_data_type, array3>);
    static_assert(std::is_same_v<decltype(s)::local_data_type, double>);
    static_assert(std::is_same_v<decltype(one)::local_data_type, samurai::StdArrayWrapper<double, 1>>);

    // The values of one cell stay an xtensor expression.
    static_assert(xt::is_xexpression<decltype(u[samurai::Cell<1, samurai::default_config::interval_t>()])>::value);

#if !defined(SAMURAI_FLUX_CONTAINER_XTENSOR) && !defined(SAMURAI_FLUX_CONTAINER_EIGEN3)
    static_assert(std::is_same_v<samurai::flux_array_t<double, 3>, array3>);
#endif
}

TEST(algebraic_array, cell_values_of_a_vector_field)
{
    const auto mesh = make_mesh();
    auto u          = samurai::make_vector_field<double, 3>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double i = static_cast<double>(cell.index);
                               const array3 a{i, 10. + i, 20. + i};

                               u[cell] = a;
                               EXPECT_EQ(u[cell](0), i);
                               EXPECT_EQ(u[cell](1), 10. + i);
                               EXPECT_EQ(u[cell](2), 20. + i);

                               u[cell] += a;
                               EXPECT_EQ(u[cell](1), 2. * (10. + i));
                               u[cell] -= 2. * a;
                               EXPECT_EQ(u[cell](1), 0.);
                               EXPECT_EQ(u[cell](2), 0.);

                               u[cell] = 0.5 * (a + a); // an array temporary

                               // The view still takes what an xtensor view takes.
                               u[cell] = u[cell] + xfixed3{1., 1., 1.};
                               EXPECT_EQ(u[cell](2), 21. + i);
                               u[cell] = xfixed3{i, 10. + i, 20. + i};
                           });

    // Nothing leaked into the neighbouring cells.
    const auto& const_u = u;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double i = static_cast<double>(cell.index);

                               const array3 read = const_u[cell];
                               expect_values(read, i, 10. + i, 20. + i);

                               array3 assigned;
                               assigned = u[cell];
                               expect_values(assigned, i, 10. + i, 20. + i);

                               // Array next to the values of a cell: an array.
                               static_assert(is_array3<decltype(read - u[cell])>);
                               static_assert(is_array3<decltype(u[cell] * 2. + read)>);
                               expect_values(read - u[cell], 0, 0, 0);
                               expect_values(u[cell] * 2. + read, 3. * i, 3. * (10. + i), 3. * (20. + i));
                           });

    // Cell to cell, and a scalar for every component.
    samurai::Cell<1, samurai::default_config::interval_t> first;
    samurai::Cell<1, samurai::default_config::interval_t> last;
    bool is_first = true;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               if (is_first)
                               {
                                   first    = cell;
                                   is_first = false;
                               }
                               last = cell;
                           });
    u[first] = u[last];
    expect_values(array3(u[first]), array3(u[last])[0], array3(u[last])[1], array3(u[last])[2]);
    u[first] = 7.;
    expect_values(array3(u[first]), 7, 7, 7);
    EXPECT_EQ(u[last](0), static_cast<double>(last.index));
}

TEST(algebraic_array, cell_values_of_a_one_component_vector_field)
{
    using array1 = samurai::StdArrayWrapper<double, 1>;

    const auto mesh = make_mesh();
    auto u          = samurai::make_vector_field<double, 1>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double i = static_cast<double>(cell.index);
                               u[cell]        = array1{i};
                               u[cell] += array1{1.};
                           });
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const array1 read = u[cell];
                               EXPECT_EQ(read[0], 1. + static_cast<double>(cell.index));
                           });
}

// A flux function the way user code writes it: values of the stencil, an
// xtensor constant, functions on arrays.
TEST(algebraic_array, flux_function_on_a_vector_field)
{
    static constexpr std::size_t dim = 1;

    const samurai::Box<double, dim> box{{0}, {1}};
    auto mesh = samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(4).max_level(4).periodic(true));
    auto u    = samurai::make_vector_field<double, 2>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               u[cell] = samurai::StdArrayWrapper<double, 2>{1. + cell.center(0), -2.};
                           });

    using field_t = decltype(u);
    using cfg     = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, field_t, field_t>;

    const xt::xtensor_fixed<double, xt::xshape<2>> velocity{1., 3.};

    samurai::FluxDefinition<cfg> flux;
    flux[0].cons_flux_function = [&](samurai::FluxValue<cfg>& f, const samurai::StencilData<cfg>&, const samurai::StencilValues<cfg>& v)
    {
        static_assert(std::is_same_v<std::decay_t<decltype(v[0])>, samurai::StdArrayWrapper<double, 2>>);
        // velocity * |uL|, written with a detour through the right value
        const auto jump = v[1] - v[0];
        f               = velocity * abs(v[1] - jump) + 0. * sqrt(abs(jump));
    };
    auto scheme = samurai::make_flux_based_scheme(flux);

    auto du = samurai::make_vector_field<double, 2>("du", mesh);
    du      = scheme(u);

    const double h = mesh.cell_length(4);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               // Upwind difference of velocity * |u|: u(1) is constant.
                               const bool first      = cell.indices[0] == 0;
                               const double expected = first ? (h - 1.) / h : 1.;
                               EXPECT_NEAR(du[cell](0), expected, 1e-12);
                               EXPECT_NEAR(du[cell](1), 0., 1e-12);
                           });
}
