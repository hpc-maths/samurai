#include "samurai/reconstruction.hpp"
#include "samurai/interval.hpp"

namespace samurai
{

    template class prediction_map<1>;
    template class prediction_map<2>;
    template class prediction_map<3>;

    template class reconstruction_op_<1, default_config::interval_t>;
    template class reconstruction_op_<2, default_config::interval_t>;
    template class reconstruction_op_<3, default_config::interval_t>;

} // namespace samurai
