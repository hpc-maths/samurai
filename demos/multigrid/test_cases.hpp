#pragma once
#include <functional>
#include <samurai/cell_array.hpp>

template <std::size_t dim>
class TestCase
{
public:
    using coords = xt::xtensor_fixed<double, xt::xshape<dim>>;
    using scalar_function = std::function<double(const coords&)>;

    virtual bool solution_is_known() { return false; }
    virtual scalar_function solution() { return nullptr; }
    virtual int solution_poly_degree() { return -1; }

    virtual scalar_function source() = 0;
    virtual int source_poly_degree()
    {
        int solution_degree = solution_poly_degree();
        return solution_degree < 0 ? -1 : std::max(solution_degree - 2, 0);
    }

    virtual scalar_function dirichlet()
    {
        if (solution_is_known())
        {
            return solution();
        }
        return [](const coords&) { return 0.0; };
    }

    virtual ~TestCase() {}
};


/**
 * The solution is a polynomial function.
 * Homogeneous Dirichlet b.c.
*/
template <std::size_t dim>
class PolynomialTestCase : public TestCase<dim>
{
    using scalar_function = typename TestCase<dim>::scalar_function;

    bool solution_is_known() override { return true; }

    scalar_function solution() override
    {
        if constexpr(dim == 1)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                return x * (1 - x);
            };
        }
        else if constexpr(dim == 2)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                return x * (1 - x) * y*(1 - y);
            };
        }
        else if constexpr(dim == 3)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                return x * (1 - x)*y*(1 - y)*z*(1 - z);
            };
        }
    }

    int solution_poly_degree() override 
    { 
        return static_cast<int>(pow(2, dim)); 
    }

    scalar_function source() override
    {
        if constexpr(dim == 1)
        {
            return [](const auto&) 
            { 
                return 2.0; 
            };
        }
        else if constexpr(dim == 2)
        {
            return [](const auto& coord) 
            { 
                const auto& x = coord[0];
                const auto& y = coord[1];
                return 2 * (y*(1 - y) + x * (1 - x));
            };
        }
        else if constexpr(dim == 3)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                return 2 * ((y*(1 - y)*z*(1 - z) + x * (1 - x)*z*(1 - z) + x * (1 - x)*y*(1 - y)));
            };
        }
    }
};


/**
 * The solution is an exponential function.
 * Non-homogeneous Dirichlet b.c.
*/
template <std::size_t dim>
class ExponentialTestCase : public TestCase<dim>
{
    using scalar_function = typename TestCase<dim>::scalar_function;

    bool solution_is_known() override { return true; }

    scalar_function solution() override
    {
        if constexpr(dim == 1)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                return exp(x);
            };
        }
        else if constexpr(dim == 2)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                return exp(x*y*y);
            };
        }
        else if constexpr(dim == 3)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                return exp(x*y*y*z*z*z);
            };
        }
    }

    scalar_function source() override
    {
        if constexpr(dim == 1)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                return -std::exp(x);
            };
        }
        else if constexpr(dim == 2)
        {
            return [](const auto& coord) 
            { 
                const auto& x = coord[0];
                const auto& y = coord[1];
                return (-pow(y, 4) - 2 * x*(1 + 2 * x*y*y))*exp(x*y*y);
            };
        }
        else if constexpr(dim == 3)
        {
            return [](const auto& coord) 
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                return -(pow(y, 4)*pow(z, 6) + 2 * x*pow(z, 3) + 4 * x*x*y*y*pow(z, 6) + 6 * x*y*y*z + 9 * x*x*pow(y, 4)*pow(z, 4))*exp(x*y*y*z*z*z);
            };
        }
    }
};