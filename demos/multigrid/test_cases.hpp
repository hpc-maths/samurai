#pragma once
#include <samurai/bc.hpp>
#include <samurai/cell_array.hpp>

template <class Field>
class TestCase
{
  public:

    static constexpr std::size_t dim = Field::dim;
    using cell_t                     = typename Field::cell_t;
    using coords_t                   = typename cell_t::coords_t;

    using boundary_cond_t  = typename samurai::FunctionBc<Field>::function_t;
    using field_value_t    = typename samurai::FunctionBc<Field>::value_t;
    using field_function_t = std::function<field_value_t(const coords_t&)>;

    virtual bool solution_is_known()
    {
        return false;
    }

    virtual field_function_t solution()
    {
        return nullptr;
    }

    virtual int solution_poly_degree()
    {
        return -1;
    }

    virtual field_function_t source() = 0;

    virtual int source_poly_degree()
    {
        int solution_degree = solution_poly_degree();
        return solution_degree < 0 ? -1 : std::max(solution_degree - 2, 0);
    }

    virtual boundary_cond_t dirichlet()
    {
        if (solution_is_known())
        {
            return [&](const auto&, const auto& coords)
            {
                return solution()(coords);
            };
        }
        return [](const cell_t&, const coords_t&)
        {
            if constexpr (Field::size == 1)
            {
                return 0.;
            }
            else
            {
                field_value_t zero;
                zero.fill(0);
                return zero;
            }
        };
    }

    virtual boundary_cond_t neumann()
    {
        assert(false && "Neumann not implemented for this test case");
        return nullptr;
    }

    virtual ~TestCase()
    {
    }
};

/**
 * The solution is a polynomial function.
 * Homogeneous Dirichlet b.c.
 */
template <class Field>
class PolynomialTestCase : public TestCase<Field>
{
    static constexpr std::size_t dim = Field::dim;
    using field_value_t              = typename TestCase<Field>::field_value_t;
    using field_function_t           = typename TestCase<Field>::field_function_t;
    using boundary_cond_t            = typename TestCase<Field>::boundary_cond_t;
    using coords_t                   = typename TestCase<Field>::coords_t;
    using cell_t                     = typename TestCase<Field>::cell_t;

    bool solution_is_known() override
    {
        return true;
    }

    field_function_t solution() override
    {
        if constexpr (dim == 1)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                return x * (1 - x);
            };
        }
        else if constexpr (dim == 2)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double value  = x * (1 - x) * y * (1 - y);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
        else if constexpr (dim == 3)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                double value  = x * (1 - x) * y * (1 - y) * z * (1 - z);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
    }

    int solution_poly_degree() override
    {
        return static_cast<int>(pow(2, dim));
    }

    boundary_cond_t dirichlet() override
    {
        return [](const cell_t&, const coords_t&)
        {
            if constexpr (Field::size == 1)
            {
                return 0.;
            }
            else
            {
                field_value_t zero;
                zero.fill(0);
                return zero;
            }
        };
    }

    boundary_cond_t neumann() override
    {
        if constexpr (dim == 1)
        {
            return [](const auto&, const coords_t& coord)
            {
                const auto& x = coord[0];
                if (x == 0 || x == 1)
                {
                    return -1;
                }
                else
                {
                    assert(false);
                    return 0;
                }
            };
        }
        else if constexpr (dim == 2)
        {
            return [](const auto&, const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double value  = 0.;
                if (x == 0 || x == 1)
                {
                    value = y * (y - 1);
                }
                else if (y == 0 || y == 1)
                {
                    value = x * (x - 1);
                }
                else
                {
                    assert(false);
                }
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
        else if constexpr (dim == 3)
        {
            assert(false && "Neumann not implemented for this test case");
            return nullptr;
        }
    }

    field_function_t source() override
    {
        if constexpr (dim == 1)
        {
            return [](const auto&)
            {
                return 2.;
            };
        }
        else if constexpr (dim == 2)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double value  = 2 * (y * (1 - y) + x * (1 - x));
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
        else if constexpr (dim == 3)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                double value  = 2 * ((y * (1 - y) * z * (1 - z) + x * (1 - x) * z * (1 - z) + x * (1 - x) * y * (1 - y)));
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
    }
};

/**
 * The solution is an exponential function.
 * Non-homogeneous Dirichlet b.c.
 */
template <class Field>
class ExponentialTestCase : public TestCase<Field>
{
    static constexpr std::size_t dim = Field::dim;
    using field_value_t              = typename TestCase<Field>::field_value_t;
    using field_function_t           = typename TestCase<Field>::field_function_t;
    using boundary_cond_t            = typename TestCase<Field>::boundary_cond_t;

    bool solution_is_known() override
    {
        return true;
    }

    field_function_t solution() override
    {
        if constexpr (dim == 1)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                return exp(x);
            };
        }
        else if constexpr (dim == 2)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double value  = exp(x * y * y);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
        else if constexpr (dim == 3)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                double value  = exp(x * y * y * z * z * z);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
    }

    field_function_t source() override
    {
        if constexpr (dim == 1)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                return -std::exp(x);
            };
        }
        else if constexpr (dim == 2)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double value  = (-pow(y, 4) - 2 * x * (1 + 2 * x * y * y)) * exp(x * y * y);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
        else if constexpr (dim == 3)
        {
            return [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                const auto& z = coord[2];
                double value  = -(pow(y, 4) * pow(z, 6) + 2 * x * pow(z, 3) + 4 * x * x * y * y * pow(z, 6) + 6 * x * y * y * z
                                 + 9 * x * x * pow(y, 4) * pow(z, 4))
                             * exp(x * y * y * z * z * z);
                if constexpr (Field::size == 1)
                {
                    return value;
                }
                else
                {
                    field_value_t values;
                    values.fill(value);
                    return values;
                }
            };
        }
    }
};
