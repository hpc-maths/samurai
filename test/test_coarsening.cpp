#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <xtensor/xmath.hpp>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/pred_and_proj.hpp>

#include "test_common.hpp"

class CoarseningTest
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, int>> {
};

std::string StringParamTestSuffix(
    const testing::TestParamInfo<std::tuple<std::size_t, std::size_t, int>> &info)
{
    auto param = info.param;
    std::stringstream out;
    out << std::get<0>(param) << "_" << std::get<1>(param) << "_"
        << std::get<2>(param);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(
    CoarseningTestNames, CoarseningTest,
    ::testing::Combine(::testing::Range<std::size_t>(1, 5), ::testing::Range<std::size_t>(2, 8),
                       ::testing::Values(1e2, 1e3, 1e4)),
    StringParamTestSuffix);

template<class Config>
auto get_init_field_1d(samurai::MRMesh<Config> &mesh, std::size_t test_case)
{
    double PI = xt::numeric_constants<double>::PI;
    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0);

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        auto x = cell.center(0);

        switch (test_case)
        {
        case 1:
            u[cell] = exp(-50.0 * x * x);
            break;
        case 2:
            u[cell] = 1 - sqrt(abs(sin(PI / 2 * x)));
            break;
        case 3:
            u[cell] = 1 - tanh(50.0 * abs(x));
            break;
        case 4:
            u[cell] = 0.5 - abs(x);
            break;
        default:
            break;
        }
    });
    return u;
}

template<class Config>
auto get_init_field_2d(samurai::MRMesh<Config> &mesh, std::size_t test_case)
{
    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0);

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        auto x = cell.center(0);
        auto y = cell.center(1);
        double radius = .2;
        double xcenter = 0, ycenter = 0;

        switch (test_case)
        {
        case 1:
            if ((x >= -.25 && x <= .25) && (y >= -.25 && y <= .25))
                u[cell] = 1;
            else
                u[cell] = 0;
            break;
        case 2:
            if (((x - xcenter) * (x - xcenter) +
                 (y - ycenter) * (y - ycenter)) <= radius * radius)
                u[cell] = 1;
            else
                u[cell] = 0;
            break;
        case 3:
            u[cell] = exp(-50 * (x * x + y * y));
            break;
        case 4:
            u[cell] = tanh(50 * (fabs(x) + fabs(y))) - 1;
            break;
        }
    });
    return u;
}

TEST_P(CoarseningTest, 1D)
{
    std::size_t test_case = std::get<0>(GetParam());
    std::size_t init_level = std::get<1>(GetParam());
    double eps = 1. / std::get<2>(GetParam());

    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({-1}, {1});
    using mesh_t = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    mesh_t mesh{box, 1, init_level};

    auto u = get_init_field_1d(mesh, test_case);

    auto update_bc = [](const auto& /*u*/, std::size_t /*level*/){};

    for (std::size_t i = 0; i < init_level; ++i)
    {
        samurai::coarsening(u, update_bc, eps, i);
    }

    for (std::size_t level1 = init_level; level1 != std::size_t(-1); --level1)
    {
        for (std::size_t level2 = level1 - 1; level2 != std::size_t(-1); --level2)
        {
            auto expr = samurai::intersection(mesh[mesh_id_t::cells][level1],
                                              mesh[mesh_id_t::cells][level2])
                            .on(level1);
            expr([](auto, auto)
            {
                RC_ASSERT(false);
            });
        }
    }
}

TEST_P(CoarseningTest, 2D)
{
    std::size_t test_case = std::get<0>(GetParam());
    std::size_t init_level = std::get<1>(GetParam());
    double eps = 1. / std::get<2>(GetParam());

    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    using mesh_t = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    mesh_t mesh{box, 1, init_level};

    auto u = get_init_field_2d(mesh, test_case);

    auto update_bc = [](const auto& /*u*/, std::size_t /*level*/){};

    for (std::size_t i = 0; i < init_level; ++i)
    {
        samurai::coarsening(u, update_bc, eps, i);
    }

    for (std::size_t level1 = init_level; level1 != std::size_t(-1); --level1)
    {
        for (std::size_t level2 = level1 - 1; level2 != std::size_t(-1); --level2)
        {
            auto expr = samurai::intersection(mesh[mesh_id_t::cells][level1],
                                              mesh[mesh_id_t::cells][level2])
                        .on(level1);
            expr([](auto, auto)
            {
                RC_ASSERT(false);
            });
        }
    }
}
