#include <algorithm> // for stable_sort
#include <cstddef>   // size_t
#include <numeric>   // for iota
#include <utility>   // pair
#include <vector>
#include <xtensor/containers/xfixed.hpp> // for coord_type

#include "samurai/array_of_interval_and_point.hpp"
#include "samurai/interval.hpp"
#include "samurai/samurai_config.hpp"

namespace samurai
{
    template class ArrayOfIntervalAndPoint<default_config::interval_t, xt::xtensor_fixed<default_config::value_t, xt::xshape<1 - 1>>>;
    template class ArrayOfIntervalAndPoint<default_config::interval_t, xt::xtensor_fixed<default_config::value_t, xt::xshape<2 - 1>>>;
    template class ArrayOfIntervalAndPoint<default_config::interval_t, xt::xtensor_fixed<default_config::value_t, xt::xshape<3 - 1>>>;
} // namespace samurai
