#include "samurai/field/scalar_field.hpp"

namespace samurai
{
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float>;

    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double>;

    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double>;
    template class ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double>;
} // namespace samurai
