// Samurai Python Bindings - MRMesh class
//
// Bindings for samurai::MRMesh class (Multiresolution Mesh)
// This is a simplified MVP binding focusing on core functionality

#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/mesh_config.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Define custom Config classes that inherit mesh_config and add mesh_id_t
// This is needed because MRMesh requires mesh_id_t in its Config
template <std::size_t dim>
struct PythonMRConfig : samurai::mesh_config<dim> {
    using mesh_id_t = samurai::MRMeshId;
};

// Helper to bind common Mesh_base methods for any mesh type
template <class Mesh>
void bind_mesh_base_common_methods(py::class_<Mesh>& cls) {
    using namespace samurai;

    // nb_cells methods - use lambdas to avoid overload resolution issues
    cls.def("nb_cells",
        [](const Mesh& mesh) -> std::size_t {
            return mesh.nb_cells();
        },
        "Total number of cells in the mesh"
    );

    cls.def("nb_cells",
        [](const Mesh& mesh, std::size_t level) -> std::size_t {
            return mesh.nb_cells(level);
        },
        py::arg("level"),
        "Number of cells at a given refinement level"
    );

    // Level properties
    cls.def_property("min_level",
        [](const Mesh& mesh) -> std::size_t {
            return mesh.min_level();
        },
        [](Mesh& mesh, std::size_t level) -> Mesh& {
            mesh.min_level() = level;
            return mesh;
        },
        "Minimum refinement level (read/write)"
    );

    cls.def_property("max_level",
        [](const Mesh& mesh) -> std::size_t {
            return mesh.max_level();
        },
        [](Mesh& mesh, std::size_t level) -> Mesh& {
            mesh.max_level() = level;
            return mesh;
        },
        "Maximum refinement level (read/write)"
    );

    // Configuration properties
    cls.def_property_readonly("graduation_width",
        &Mesh::graduation_width,
        "AMR graduation width"
    );

    cls.def_property_readonly("ghost_width",
        &Mesh::ghost_width,
        "Ghost width (for stencil operations)"
    );

    cls.def_property_readonly("max_stencil_radius",
        &Mesh::max_stencil_radius,
        "Maximum stencil radius"
    );

    // Cell lengths
    cls.def("cell_length",
        &Mesh::cell_length,
        py::arg("level"),
        "Length of a cell at given refinement level"
    );

    cls.def_property_readonly("min_cell_length",
        &Mesh::min_cell_length,
        "Minimum cell length in the mesh"
    );

    // Periodicity
    cls.def("is_periodic",
        [](const Mesh& mesh) -> bool {
            return mesh.is_periodic();
        },
        "Check if mesh is periodic in any direction"
    );

    cls.def("is_periodic",
        [](const Mesh& mesh, std::size_t d) -> bool {
            return mesh.is_periodic(d);
        },
        py::arg("direction"),
        "Check if mesh is periodic in a specific direction"
    );

    cls.def_property_readonly("periodicity",
        &Mesh::periodicity,
        "Array of periodicity flags for each direction"
    );

    // Origin point and scaling
    cls.def_property_readonly("origin_point",
        &Mesh::origin_point,
        "Origin point of the mesh domain"
    );

    cls.def_property_readonly("scaling_factor",
        &Mesh::scaling_factor,
        "Scaling factor for coordinates"
    );

    // String representation
    cls.def("__repr__",
        [](const Mesh& mesh) {
            std::ostringstream oss;
            oss << "MRMesh" << Mesh::dim << "D(";
            oss << "min_level=" << mesh.min_level();
            oss << ", max_level=" << mesh.max_level();
            oss << ", nb_cells=" << mesh.nb_cells();
            oss << ")";
            return oss.str();
        });

    cls.def("__str__",
        [](const Mesh& mesh) {
            std::ostringstream oss;
            oss << "MRMesh" << Mesh::dim << "D";
            oss << " [L" << mesh.min_level() << "-" << mesh.max_level() << "]";
            oss << " [" << mesh.nb_cells() << " cells]";
            return oss.str();
        });
}

// Template function to bind MRMesh for a specific dimension
template <std::size_t dim>
void bind_mr_mesh(py::module_& m, const std::string& name) {
    // Define the configuration type with mesh_id_t
    using Config = PythonMRConfig<dim>;
    using Mesh = samurai::MRMesh<Config>;
    using Box = samurai::Box<double, dim>;

    auto cls = py::class_<Mesh>(m, name.c_str(), R"pbdoc(
        Multiresolution Mesh (MRMesh)

        Adaptive mesh refinement mesh with multiresolution analysis capabilities.

        Note: Creating MRMesh is computationally intensive. Use small level ranges for testing.

        Examples
        --------
        >>> import samurai as sam
        >>> box = sam.Box1D([0.], [1.])
        >>> config = sam.MeshConfig1D()
        >>> config.min_level = 0
        >>> config.max_level = 2
        >>> mesh = sam.MRMesh1D(box, config)

        Attributes
        ----------
        min_level : int
            Minimum refinement level
        max_level : int
            Maximum refinement level
        nb_cells : int
            Total number of cells
        graduation_width : int
            AMR graduation width
        ghost_width : int
            Ghost width for stencil operations
    )pbdoc");

    // Constructor factory function - convert MeshConfig to our internal Config type
    cls.def(py::init([](const Box& box, const samurai::mesh_config<dim>& user_config) {
        // Copy user config values to our Config type
        Config config;
        config.min_level() = user_config.min_level();
        config.max_level() = user_config.max_level();
        config.start_level() = user_config.start_level();
        config.graduation_width() = user_config.graduation_width();
        config.max_stencil_radius() = user_config.max_stencil_radius();
        config.scaling_factor() = user_config.scaling_factor();
        config.approx_box_tol() = user_config.approx_box_tol();
        config.periodic() = user_config.periodic();

        return Mesh(box, config);
    }),
        py::arg("box"),
        py::arg("config"),
        "Create MRMesh from Box and MeshConfig"
    );

    // Bind all common methods from Mesh_base
    bind_mesh_base_common_methods<Mesh>(cls);

    // Dimension property (read-only)
    cls.def_property_readonly("dim",
        [](const Mesh&) { return dim; },
        "Dimension of the mesh"
    );
}

// Module initialization function for MRMesh bindings
void init_mesh_bindings(py::module_& m) {
    // Bind MRMesh classes for dimensions 1, 2, 3
    bind_mr_mesh<1>(m, "MRMesh1D");
    bind_mr_mesh<2>(m, "MRMesh2D");
    bind_mr_mesh<3>(m, "MRMesh3D");

    // Also expose them in a submodule for better organization
    py::module_ mesh = m.def_submodule("mesh", "Mesh classes");
    mesh.attr("MRMesh1D") = m.attr("MRMesh1D");
    mesh.attr("MRMesh2D") = m.attr("MRMesh2D");
    mesh.attr("MRMesh3D") = m.attr("MRMesh3D");
}
