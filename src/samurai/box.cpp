#include "samurai/box.hpp"

namespace samurai
{

    template class Box<default_config::value_t, 1>;
    template class Box<default_config::value_t, 2>;
    template class Box<default_config::value_t, 3>;

} // namespace samurai
