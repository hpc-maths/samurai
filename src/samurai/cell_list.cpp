#include "cell_list.hpp"

namespace samurai
{

    template class CellList<1, default_config::interval_t, default_config::max_level>;
    template class CellList<2, default_config::interval_t, default_config::max_level>;
    template class CellList<3, default_config::interval_t, default_config::max_level>;

} // namespace samurai
