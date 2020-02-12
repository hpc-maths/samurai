#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/mr/coarsening.hpp>
#include <mure/mr/mesh.hpp>
#include <mure/mr/mr_config.hpp>
#include <mure/mr/pred_and_proj.hpp>

#include "test_common.hpp"

class CoarseningTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {
};

std::string StringParamTestSuffix(
    const testing::TestParamInfo<std::tuple<int, int, int>> &info)
{
    auto param = info.param;
    std::stringstream out;
    out << std::get<0>(param) << "_" << std::get<1>(param) << "_"
        << std::get<2>(param);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(
    CoarseningTestNames, CoarseningTest,
    ::testing::Combine(::testing::Range(1, 5), ::testing::Range(2, 11),
                       ::testing::Values(1e1, 1e2, 1e3, 1e4, 1e5)),
    StringParamTestSuffix);

template<class Config>
auto get_init_field_1d(mure::Mesh<Config> &mesh, std::size_t test_case)
{
    mure::Field<Config> u("u", mesh);
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        switch (test_case)
        {
        case 1:
            u[cell] = exp(-50.0 * x * x);
            break;
        case 2:
            u[cell] = 1 - sqrt(abs(sin(M_PI / 2 * x)));
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
auto get_init_field_2d(mure::Mesh<Config> &mesh, std::size_t test_case)
{
    mure::Field<Config> u("u", mesh);
    u.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];
        double radius = .2;
        double xcenter = 0, ycenter = 0;

        switch (test_case)
        {
        case 1:
            if ((x >= -.25 and x <= .25) and (y >= -.25 and y <= .25))
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
    using Config = mure::MRConfig<dim>;

    mure::Box<double, dim> box({-1}, {1});
    mure::Mesh<Config> mesh{box, init_level};

    auto u = get_init_field_1d(mesh, test_case);

    for (std::size_t i = 0; i < init_level; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        mure::mr_projection(u);
        mure::coarsening(detail, u, eps, i);
    }

    for (int level1 = init_level; level1 > 0; --level1)
    {
        for (int level2 = level1 - 1; level2 > 0; --level2)
        {
            auto expr = mure::intersection(mesh[mure::MeshType::cells][level1],
                                           mesh[mure::MeshType::cells][level2])
                            .on(level1);
            expr([](auto &, auto &, auto &) { RC_ASSERT(false); });
        }
    }
}

TEST_P(CoarseningTest, 2D)
{
    std::size_t test_case = std::get<0>(GetParam());
    std::size_t init_level = std::get<1>(GetParam());
    double eps = 1. / std::get<2>(GetParam());

    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;

    mure::Box<double, dim> box({-1, -1}, {1, 1});
    mure::Mesh<Config> mesh{box, init_level};

    auto u = get_init_field_2d(mesh, test_case);

    for (std::size_t i = 0; i < init_level; ++i)
    {
        mure::Field<Config> detail{"detail", mesh};
        detail.array().fill(0);
        mure::mr_projection(u);
        mure::coarsening(detail, u, eps, i);
    }

    for (int level1 = init_level; level1 > 0; --level1)
    {
        for (int level2 = level1 - 1; level2 > 0; --level2)
        {
            auto expr = mure::intersection(mesh[mure::MeshType::cells][level1],
                                           mesh[mure::MeshType::cells][level2])
                            .on(level1);
            expr([](auto &, auto &, auto &) { RC_ASSERT(false); });
        }
    }
}
