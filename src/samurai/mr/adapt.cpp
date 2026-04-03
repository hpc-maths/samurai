#include "samurai/mr/adapt.hpp"

namespace samurai
{
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, float>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, float>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, float>>;

    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, double>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, double>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, double>>;

    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, long double>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, long double>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, long double>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, float>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, float>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, float>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, double>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, double>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, double>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<1>>, long double>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<2>>, long double>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), ScalarField<MRMesh<mesh_config<3>>, long double>>;

    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, float, 1, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, float, 1, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, float, 2, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, float, 2, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, float, 3, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, float, 3, true>>;

    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, double, 1, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, double, 1, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, double, 2, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, double, 2, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, double, 3, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, double, 3, true>>;

    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, long double, 1, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, long double, 1, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, long double, 2, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, long double, 2, true>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, long double, 3, false>>;
    template class Adapt<false, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, long double, 3, true>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, float, 1, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, float, 1, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, float, 2, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, float, 2, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, float, 3, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, float, 3, true>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, double, 1, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, double, 1, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, double, 2, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, double, 2, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, double, 3, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, double, 3, true>>;

    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, long double, 1, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<1>>, long double, 1, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, long double, 2, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<2>>, long double, 2, true>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, long double, 3, false>>;
    template class Adapt<true, decltype(default_config::default_prediction_fn), VectorField<MRMesh<mesh_config<3>>, long double, 3, true>>;
} // namespace samurai
