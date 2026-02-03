#pragma once
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/views/xview.hpp>

#include "interval.hpp"
#include "samurai_config.hpp"

using namespace xt::placeholders;

namespace samurai
{
    /**
     * Stores the triplet (level, i, index)
     */
    template <std::size_t dim, class TInterval>
    struct MeshInterval
    {
        using interval_t = TInterval;
        using coord_type = xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>>;

        std::size_t level;
        interval_t i;
        coord_type index;

        MeshInterval()
            : level(0)
        {
        }

        MeshInterval(std::size_t _level)
            : level(_level)
        {
        }

        MeshInterval(std::size_t _level, const interval_t& _i, const coord_type& _index)
            : level(_level)
            , i(_i)
            , index(_index)
        {
        }
    };

    extern template class MeshInterval<1, default_config::interval_t>;
    extern template class MeshInterval<2, default_config::interval_t>;
    extern template class MeshInterval<3, default_config::interval_t>;

    template <std::size_t dim, class TInterval>
    MeshInterval<dim, TInterval>
    operator+(const MeshInterval<dim, TInterval>& mi, const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim>>& translate)
    {
        return (mi.level, mi.i + translate(0), mi.index + xt::view(translate, xt::range(1, _)));
    }

}
