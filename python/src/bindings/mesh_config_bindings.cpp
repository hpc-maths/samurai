// Samurai Python Bindings - MeshConfig class
//
// Bindings for samurai::mesh_config class
// Provides fluent interface for mesh configuration

#include <samurai/mesh_config.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Type aliases for mesh_config with default template parameters
using MeshConfig1D = samurai::mesh_config<1>;
using MeshConfig2D = samurai::mesh_config<2>;
using MeshConfig3D = samurai::mesh_config<3>;

// Helper to bind MeshConfig methods that return *this for chaining
template <std::size_t dim, class Config>
void bind_mesh_config_common_methods(py::class_<Config>& cls) {
    using namespace samurai;

    // Min level
    cls.def_property("min_level",
        [](const Config& cfg) { return cfg.min_level(); },
        [](Config& cfg, std::size_t level) {
            cfg.min_level(level);
            return cfg;
        },
        "Minimum refinement level (read/write, returns config for chaining)"
    );

    // Max level
    cls.def_property("max_level",
        [](const Config& cfg) { return cfg.max_level(); },
        [](Config& cfg, std::size_t level) {
            cfg.max_level(level);
            return cfg;
        },
        "Maximum refinement level (read/write, returns config for chaining)"
    );

    // Start level
    cls.def_property("start_level",
        [](const Config& cfg) { return cfg.start_level(); },
        [](Config& cfg, std::size_t level) {
            cfg.start_level(level);
            return cfg;
        },
        "Starting refinement level (read/write, returns config for chaining)"
    );

    // Graduation width
    cls.def_property("graduation_width",
        [](const Config& cfg) { return cfg.graduation_width(); },
        [](Config& cfg, std::size_t width) {
            cfg.graduation_width(width);
            return cfg;
        },
        "Graduation width for AMR (read/write, returns config for chaining)"
    );

    // Max stencil radius
    cls.def_property("max_stencil_radius",
        [](const Config& cfg) { return cfg.max_stencil_radius(); },
        [](Config& cfg, int radius) {
            cfg.max_stencil_radius(radius);
            return cfg;
        },
        "Maximum stencil radius (read/write, returns config for chaining)"
    );

    // Max stencil size (derived from radius)
    cls.def_property("max_stencil_size",
        [](const Config& cfg) { return cfg.max_stencil_size(); },
        [](Config& cfg, int size) {
            cfg.max_stencil_size(size);
            return cfg;
        },
        "Maximum stencil size (read/write, returns config for chaining)"
    );

    // Scaling factor
    cls.def_property("scaling_factor",
        [](const Config& cfg) { return cfg.scaling_factor(); },
        [](Config& cfg, double factor) {
            cfg.scaling_factor(factor);
            return cfg;
        },
        "Scaling factor for coordinates (read/write, returns config for chaining)"
    );

    // Approx box tolerance
    cls.def_property("approx_box_tol",
        [](const Config& cfg) { return cfg.approx_box_tol(); },
        [](Config& cfg, double tol) {
            cfg.approx_box_tol(tol);
            return cfg;
        },
        "Approximation tolerance for box (read/write, returns config for chaining)"
    );

    // Ghost width (read-only)
    cls.def_property_readonly("ghost_width",
        &Config::ghost_width,
        "Ghost width (read-only, computed from stencil)"
    );

    // Periodic (scalar - set all directions)
    cls.def("set_periodic",
        [](Config& cfg, bool periodic) -> Config& {
            cfg.periodic(periodic);
            return cfg;
        },
        py::arg("periodic"),
        "Set periodicity in all directions (returns config for chaining)"
    );

    // Periodic (array - per direction)
    cls.def("set_periodic_per_direction",
        [](Config& cfg, const std::array<bool, dim>& periodic) -> Config& {
            cfg.periodic(periodic);
            return cfg;
        },
        py::arg("periodic"),
        "Set periodicity per direction (returns config for chaining)"
    );

    cls.def("get_periodic",
        [](const Config& cfg, std::size_t i) {
            if (i >= dim) {
                throw std::out_of_range("Periodic index out of range");
            }
            return cfg.periodic(i);
        },
        py::arg("direction"),
        "Get periodicity in specific direction"
    );

    // String representation
    cls.def("__repr__",
        [](const Config& cfg) {
            constexpr std::size_t d = Config::dim;
            std::ostringstream oss;
            oss << "MeshConfig" << d << "D(";
            oss << "min_level=" << cfg.min_level();
            oss << ", max_level=" << cfg.max_level();
            oss << ", start_level=" << cfg.start_level();
            oss << ", graduation_width=" << cfg.graduation_width();
            oss << ")";
            return oss.str();
        });

    cls.def("__str__",
        [](const Config& cfg) {
            constexpr std::size_t d = Config::dim;
            std::ostringstream oss;
            oss << "MeshConfig" << d << "D";
            oss << " [min=" << cfg.min_level();
            oss << ", max=" << cfg.max_level();
            oss << ", start=" << cfg.start_level();
            oss << "]";
            return oss.str();
        });
}

// Template function to bind MeshConfig for any dimension
template <std::size_t dim>
void bind_mesh_config(py::module_& m, const std::string& name) {
    using Config = samurai::mesh_config<dim>;

    auto cls = py::class_<Config>(m, name.c_str(), R"pbdoc(
        Mesh configuration class with fluent interface.

        Used to configure mesh parameters for AMR/MR algorithms.

        Parameters
        ----------
        None - creates default configuration

        Examples
        --------
        >>> import samurai as sam
        >>> config = sam.MeshConfig2D()
        >>> config.min_level = 2
        >>> config.max_level = 6
        >>> # Or use method chaining
        >>> config = sam.MeshConfig2D().min_level(2).max_level(6)

        Attributes
        ----------
        min_level : int
            Minimum refinement level (default: 0)
        max_level : int
            Maximum refinement level (default: 6)
        start_level : int
            Starting refinement level (default: 6)
        graduation_width : int
            AMR graduation width (default: depends on config)
        max_stencil_radius : int
            Maximum stencil radius
        max_stencil_size : int
            Maximum stencil size (2 * radius)
        scaling_factor : float
            Coordinate scaling factor
        approx_box_tol : float
            Approximation tolerance for box
        ghost_width : int (read-only)
            Ghost width, computed from stencil
    )pbdoc");

    // Constructor
    cls.def(py::init<>(), "Create default mesh configuration");

    // Bind all common methods
    bind_mesh_config_common_methods<dim>(cls);

    // Dimension property (read-only)
    cls.def_property_readonly("dim",
        [](const Config&) { return Config::dim; },
        "Dimension of the mesh configuration"
    );
}

// Module initialization function for MeshConfig bindings
void init_mesh_config_bindings(py::module_& m) {
    // Bind MeshConfig classes for dimensions 1, 2, 3
    bind_mesh_config<1>(m, "MeshConfig1D");
    bind_mesh_config<2>(m, "MeshConfig2D");
    bind_mesh_config<3>(m, "MeshConfig3D");

    // Also expose them in a submodule for better organization
    py::module_ config = m.def_submodule("config", "Mesh configuration classes");
    config.attr("MeshConfig1D") = m.attr("MeshConfig1D");
    config.attr("MeshConfig2D") = m.attr("MeshConfig2D");
    config.attr("MeshConfig3D") = m.attr("MeshConfig3D");
}
