#include "samurai/mr/adapt.hpp"

namespace samurai
{
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float>>;

    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double>>;

    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         ScalarField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double>>;

    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float, 1, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float, 1, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float, 2, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float, 2, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float, 3, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float, 3, true>>;

    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double, 1, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double, 1, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double, 2, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double, 2, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double, 3, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double, 3, true>>;

    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double, 1, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double, 1, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double, 2, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double, 2, true>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double, 3, false>>;
    template class Adapt<false,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double, 3, true>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float, 1, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, float, 1, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float, 2, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, float, 2, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float, 3, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, float, 3, true>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double, 1, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, double, 1, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double, 2, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, double, 2, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double, 3, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, double, 3, true>>;

    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double, 1, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<1>, MRMeshId>>, long double, 1, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double, 2, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<2>, MRMeshId>>, long double, 2, true>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double, 3, false>>;
    template class Adapt<true,
                         decltype(default_config::default_prediction_fn),
                         VectorField<MRMesh<complete_mesh_config<mesh_config<3>, MRMeshId>>, long double, 3, true>>;
} // namespace samurai
