#pragma once

#include <mure/operators_base.hpp>

namespace mure
{

    template<class TInterval>
    class to_coarsen_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_coarsen_mr_op)

        template<class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, double max_det, T2 &tag, double eps) const
        {
            
            auto mask = (xt::abs(detail(level, 2*i  ,   2*j))/max_det < eps) and
                        (xt::abs(detail(level, 2*i+1,   2*j))/max_det < eps) and
                        (xt::abs(detail(level, 2*i  , 2*j+1))/max_det < eps) and
                        (xt::abs(detail(level, 2*i+1, 2*j+1))/max_det < eps);
            /*
            auto mask = 0.25 * (xt::abs(detail(level, 2*i  ,   2*j)) +
                        xt::abs(detail(level, 2*i+1,   2*j)) +
                        xt::abs(detail(level, 2*i  , 2*j+1)) +
                        xt::abs(detail(level, 2*i+1, 2*j+1))) < eps;
            */          
                        
                        
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

    
    
    
    template<class TInterval>
    class to_refine_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_refine_mr_op)

        template<class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, double max_det, 
                               T2 &tag, double eps, std::size_t max_level) const
        {

            if (level < max_level)  {
                // One with true is sufficient.
                auto mask = (xt::abs(detail(level, 2*i  ,   2*j))/max_det > eps) or
                            (xt::abs(detail(level, 2*i+1,   2*j))/max_det > eps) or
                            (xt::abs(detail(level, 2*i  , 2*j+1))/max_det > eps) or
                            (xt::abs(detail(level, 2*i+1, 2*j+1))/max_det > eps);
                             
                xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::refine);
                xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::refine);
                xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
        
            }
        
        }
    };

    template<class... CT>
    inline auto to_refine_mr(CT &&... e)
    {
        return make_field_operator_function<to_refine_mr_op>(
            std::forward<CT>(e)...);
    }



    template<class TInterval>
    class max_detail_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(max_detail_mr_op)

        template<class T1>
        inline void operator()(Dim<2>, const T1& detail, double & max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            max_detail =
                std::max(max_detail,
                         xt::amax(xt::maximum(
                             xt::abs(detail(level + 1, ii, 2 * j)),
                             xt::abs(detail(level + 1, ii, 2 * j + 1))))[0]);

        }
    };

    template<class... CT>
    inline auto max_detail_mr(CT &&... e)
    {
        return make_field_operator_function<max_detail_mr_op>(
            std::forward<CT>(e)...);
    }



}