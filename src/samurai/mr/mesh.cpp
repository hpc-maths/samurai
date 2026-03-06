#include "samurai/mr/mesh.hpp"

namespace samurai
{
    template class MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>;
    template class MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>;
    template class MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>;
}
