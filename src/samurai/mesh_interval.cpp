#include "samurai/mesh_interval.hpp"
#include "samurai/interval.hpp"

namespace samurai
{

    template struct MeshInterval<1, default_config::interval_t>;
    template struct MeshInterval<2, default_config::interval_t>;
    template struct MeshInterval<3, default_config::interval_t>;

} // namespace samurai
