#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace samurai
{

    namespace detail
    {

        template <size_t index_size, size_t dim, size_t dim_min>
        struct NestedLoop
        {
            using index_type = xt::xtensor_fixed<int, xt::xshape<index_size>>;
            template <int value>
            using int_constant = std::integral_constant<int, value>;

            template <typename Function, int I0, int I1, int STEP = 1>
            static void run(index_type& idx, int_constant<I0> i0, int_constant<I1> i1, int_constant<STEP> step, Function&& func)
            {
                if constexpr (dim != dim_min - 1)
                {
                    for (idx[dim] = I0; idx[dim] < I1; idx[dim] += STEP)
                    {
                        NestedLoop<index_size, dim - 1, dim_min>::run(idx, i0, i1, step, std::forward<Function>(func));
                    }
                }
                else
                {
                    func(idx);
                }
            }

            template <typename Function>
            static void run(index_type& idx, int i0, int i1, Function&& func)
            {
                if constexpr (dim != dim_min - 1)
                {
                    for (idx[dim] = i0; idx[dim] != i1; ++idx[dim])
                    {
                        NestedLoop<index_size, dim - 1, dim_min>::run(idx, i0, i1, std::forward<Function>(func));
                    }
                }
                else
                {
                    func(idx);
                }
            }
        };

    }

    template <size_t index_size, size_t dim_min, size_t dim_max, typename Function, int i0, int I1>
    inline void staticNestedLoop(std::integral_constant<int, i0>, std::integral_constant<int, I1>, Function&& func)
    {
        using index_type  = typename detail::NestedLoop<index_size, dim_max - 1, dim_min>::index_type;
        using i0_constant = typename detail::NestedLoop<index_size, dim_max - 1, dim_min>::template int_constant<i0>;
        using i1_constant = typename detail::NestedLoop<index_size, dim_max - 1, dim_min>::template int_constant<I1>;
        index_type idx;
        for (size_t i = 0; i != dim_min; ++i)
        {
            idx[i] = i0;
        }
        for (size_t i = dim_max; i != index_size; ++i)
        {
            idx[i] = i0;
        }
        detail::NestedLoop<index_size, dim_max - 1, dim_min>::run(idx, i0_constant{}, i1_constant{}, std::forward<Function>(func));
    }

    template <size_t index_size, size_t dim_min, size_t dim_max, typename Function>
    inline void nestedLoop(int i0, int i1, Function&& func)
    {
        using index_type = typename detail::NestedLoop<index_size, dim_max - 1, dim_min>::index_type;
        index_type idx;
        for (size_t i = 0; i != dim_min; ++i)
        {
            idx[i] = i0;
        }
        for (size_t i = dim_max; i != index_size; ++i)
        {
            idx[i] = i0;
        }
        detail::NestedLoop<index_size, dim_max - 1, dim_min>::run(idx, i0, i1, std::forward<Function>(func));
    }

    template <size_t index_size, typename Function, int I0, int I1>
    inline void staticNestedLoop(std::integral_constant<int, I0> i0, std::integral_constant<int, I1> i1, Function&& func)
    {
        staticNestedLoop<index_size, 0, index_size>(i0, i1, std::forward<Function>(func));
    }

    template <size_t index_size, typename Function>
    inline void nestedLoop(int i0, int i1, Function&& func)
    {
        nestedLoop<index_size, 0, index_size>(i0, i1, std::forward<Function>(func));
    }

}
