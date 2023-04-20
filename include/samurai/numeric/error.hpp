#pragma once
#include "gauss_legendre.hpp"

namespace samurai
{
    /**
     * Computes the L2-error with respect to an exact solution.
     * @tparam relative_error: if true, compute the relative error instead of the absolute one.
     */
    template <bool relative_error, class Field, class Func>
    double L2_error(Field& approximate, Func&& exact)
    {
        // In FV, we want only 1 quadrature point.
        // This is equivalent to
        //       error += pow(exact(cell.center()) - approximate(cell.index), 2) * cell.length^dim;
        GaussLegendre<0> gl;

        double error_norm    = 0;
        double solution_norm = 0;
        for_each_cell(approximate.mesh(),
                      [&](const auto& cell)
                      {
                          error_norm += gl.quadrature<1>(cell,
                                                         [&](const auto& point)
                                                         {
                                                             auto e = exact(point) - approximate[cell];
                                                             double norm_square;
                                                             if constexpr (Field::size == 1)
                                                             {
                                                                 norm_square = e * e;
                                                             }
                                                             else
                                                             {
                                                                 norm_square = xt::sum(e * e)();
                                                             }
                                                             return norm_square;
                                                         });
                          if constexpr (relative_error)
                          {
                              solution_norm += gl.quadrature<1>(cell,
                                                                [&](const auto& point)
                                                                {
                                                                    auto v = exact(point);
                                                                    double v_square;
                                                                    if constexpr (Field::size == 1)
                                                                    {
                                                                        v_square = v * v;
                                                                    }
                                                                    else
                                                                    {
                                                                        v_square = xt::sum(v * v)();
                                                                    }
                                                                    return v_square;
                                                                });
                          }
                      });

        error_norm    = sqrt(error_norm);
        solution_norm = sqrt(solution_norm);
        if constexpr (relative_error)
        {
            return error_norm / solution_norm;
        }
        else
        {
            return error_norm;
        }
    }

    template <class Field, class Func>
    double L2_error(Field& approximate, Func&& exact)
    {
        return L2_error<false, Field, Func>(approximate, std::forward<Func>(exact));
    }

    template <std::size_t order>
    double compute_error_bound_hidden_constant(double h, double error)
    {
        return error / pow(h, order);
    }

    template <std::size_t order>
    double theoretical_error_bound(double hidden_constant, double h)
    {
        return hidden_constant * pow(h, order);
    }

    double convergence_order(double h1, double error1, double h2, double error2)
    {
        return log(error2 / error1) / log(h2 / h1);
    }
}
