#include "cell_array.hpp"

namespace samurai
{

    template class CellArray<1, default_config::interval_t, default_config::max_level>;
    template class CellArray<2, default_config::interval_t, default_config::max_level>;
    template class CellArray<3, default_config::interval_t, default_config::max_level>;

} // namespace samurai
