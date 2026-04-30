#include "samurai/stencil_field.hpp"

namespace samurai
{

    template class upwind_op<1, default_config::interval_t>;
    template class upwind_op<2, default_config::interval_t>;
    template class upwind_op<3, default_config::interval_t>;

    template class upwind_scalar_burgers_op<1, default_config::interval_t>;
    template class upwind_scalar_burgers_op<2, default_config::interval_t>;
    template class upwind_scalar_burgers_op<3, default_config::interval_t>;

} // namespace samurai
