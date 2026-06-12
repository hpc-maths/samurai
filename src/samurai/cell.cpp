#include "samurai/cell.hpp"

namespace samurai
{

    template struct Cell<1, default_config::interval_t>;
    template struct Cell<2, default_config::interval_t>;
    template struct Cell<3, default_config::interval_t>;

} // namespace samurai
