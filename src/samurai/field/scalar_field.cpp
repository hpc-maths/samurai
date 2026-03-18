#include "samurai/field/scalar_field.hpp"

namespace samurai
{
    template class ScalarField<MRMesh<mesh_config<1>>, float>;
    template class ScalarField<MRMesh<mesh_config<2>>, float>;
    template class ScalarField<MRMesh<mesh_config<3>>, float>;

    template class ScalarField<MRMesh<mesh_config<1>>, double>;
    template class ScalarField<MRMesh<mesh_config<2>>, double>;
    template class ScalarField<MRMesh<mesh_config<3>>, double>;

    template class ScalarField<MRMesh<mesh_config<1>>, long double>;
    template class ScalarField<MRMesh<mesh_config<2>>, long double>;
    template class ScalarField<MRMesh<mesh_config<3>>, long double>;
} // namespace samurai
