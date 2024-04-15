#include <map>

#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <samurai/storage/eigen.hpp>
#include <samurai/storage/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace samurai
{
    auto arange_xtensor(std::size_t size, std::size_t rows, std::size_t cols)
    {
        return xt::arange<double>(static_cast<double>(size)).reshape({rows, cols});
    }

    auto arange_eigen(std::size_t size, std::size_t rows, std::size_t cols)
    {
        return Eigen::VectorXd::LinSpaced(static_cast<long>(size), 0., static_cast<double>(size) - 1).reshaped<Eigen::RowMajor>(rows, cols);
    }

    template <template <class, std::size_t, bool> class container_t>
    void constructor()
    {
        container_t<double, 1, false> container_1(5);
        container_t<double, 4, false> container_2(5);
        container_t<double, 5, true> container_3(5);
    }

    TEST(storage_xtensor, constructor)
    {
        constructor<xtensor_container>();
    }

    TEST(storage_xtensor, view)
    {
        xtensor_container<double, 1, false> container_1(5);

        container_1.data() = xt::arange<double>(5);
        auto v_1           = view(container_1, {1, 3});

        EXPECT_EQ(v_1(0), 1);
        EXPECT_EQ(v_1(1), 2);

        xtensor_container<double, 4, false> container_2(5);
        container_2.data() = arange_xtensor(20, 5, 4);
        auto v_2           = view(container_2, {1, 3});
        EXPECT_EQ(v_2(0, 0), 4);
        EXPECT_EQ(v_2(0, 2), 6);
        EXPECT_EQ(v_2(1, 3), 11);

        xtensor_container<double, 4, true> container_3(5);
        container_3.data() = arange_xtensor(20, 4, 5);
        auto v_3           = view(container_3, {1, 3});
        EXPECT_EQ(v_3(0, 0), 1);
        EXPECT_EQ(v_3(2, 0), 11);
        EXPECT_EQ(v_3(3, 1), 17);
    }

    TEST(storage_eigen, view)
    {
        eigen_container<double, 1, false> container_1(5);

        container_1.data() = Eigen::VectorXd::LinSpaced(5, 0, 4);
        auto v_1           = view(container_1, {1, 3});

        EXPECT_EQ(v_1(0), 1);
        EXPECT_EQ(v_1(1), 2);

        eigen_container<double, 4, false> container_2(5);
        container_2.data() = arange_eigen(20, 5, 4);
        auto v_2           = view(container_2, {1, 3});

        EXPECT_EQ(v_2(0, 0), 4);
        EXPECT_EQ(v_2(0, 2), 6);
        EXPECT_EQ(v_2(1, 3), 11);

        eigen_container<double, 4, true> container_3(5);
        container_3.data() = arange_eigen(20, 4, 5);
        auto v_3           = view(container_3, {1, 3});
        EXPECT_EQ(v_3(0, 0), 1);
        EXPECT_EQ(v_3(2, 0), 11);
        EXPECT_EQ(v_3(3, 1), 17);
    }

}
