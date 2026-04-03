#include "samurai/field/vector_field.hpp"

namespace samurai
{
    template class VectorField<MRMesh<mesh_config<1>>, float, 1, false>;
    template class VectorField<MRMesh<mesh_config<2>>, float, 1, true>;
    template class VectorField<MRMesh<mesh_config<3>>, float, 2, false>;
    template class VectorField<MRMesh<mesh_config<1>>, float, 2, true>;
    template class VectorField<MRMesh<mesh_config<2>>, float, 3, false>;
    template class VectorField<MRMesh<mesh_config<3>>, float, 3, true>;

    template class VectorField<MRMesh<mesh_config<1>>, double, 1, false>;
    template class VectorField<MRMesh<mesh_config<2>>, double, 1, true>;
    template class VectorField<MRMesh<mesh_config<3>>, double, 2, false>;
    template class VectorField<MRMesh<mesh_config<1>>, double, 2, true>;
    template class VectorField<MRMesh<mesh_config<2>>, double, 3, false>;
    template class VectorField<MRMesh<mesh_config<3>>, double, 3, true>;

    template class VectorField<MRMesh<mesh_config<1>>, long double, 1, false>;
    template class VectorField<MRMesh<mesh_config<2>>, long double, 1, true>;
    template class VectorField<MRMesh<mesh_config<3>>, long double, 2, false>;
    template class VectorField<MRMesh<mesh_config<1>>, long double, 2, true>;
    template class VectorField<MRMesh<mesh_config<2>>, long double, 3, false>;
    template class VectorField<MRMesh<mesh_config<3>>, long double, 3, true>;
} // namespace samurai
