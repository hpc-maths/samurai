#pragma once

#include <mure/operators_base.hpp>

namespace mure
{

    template<class TInterval>
    class to_coarsen_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_coarsen_mr_op)

        template<class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, T2 &tag, double eps) const
        {
            auto mask = (detail(level, 2*i  ,   2*j) < eps) and
                        (detail(level, 2*i+1,   2*j) < eps) and
                        (detail(level, 2*i  , 2*j+1) < eps) and
                        (detail(level, 2*i+1, 2*j+1) < eps);

            xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
            xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
            xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
            xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
        }
    };

    template<class... CT>
    inline auto to_coarsen_mr(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_mr_op>(
            std::forward<CT>(e)...);
    }


}