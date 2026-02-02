#include "array_of_interval_and_point.hpp"

namespace samurai
{

    template class ArrayOfIntervalAndPoint<default_config::interval_t, typename LevelCellArray<1, default_config::interval_t>::coords_t>;
    template class ArrayOfIntervalAndPoint<default_config::interval_t, typename LevelCellArray<2, default_config::interval_t>::coords_t>;
    template class ArrayOfIntervalAndPoint<default_config::interval_t, typename LevelCellArray<3, default_config::interval_t>::coords_t>;

} // namespace samurai
