#include "samurai/mesh_interval.hpp"
#include "samurai/interval.hpp"

namespace samurai
{

    template class MeshInterval<1, default_config::interval_t>;
    template class MeshInterval<2, default_config::interval_t>;
    template class MeshInterval<3, default_config::interval_t>;

} // namespace samurai
