// Samurai Python Bindings - Multiresolution Configuration
//
// Bindings for mra_config class used in mesh adaptation

#include <samurai/arguments.hpp>
#include <samurai/mr/config.hpp>
#include <pybind11/pybind11.h>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

// ============================================================
// MRA Configuration bindings
// ============================================================

// Helper function to bind common properties with fluent interface support
void bind_mra_config_properties(py::class_<samurai::mra_config>& cls)
{
    using namespace samurai;

    // Epsilon property - tolerance for mesh adaptation
    cls.def_property("epsilon",
        [](const mra_config& cfg) {
            return cfg.epsilon();
        },
        [](mra_config& cfg, double eps) {
            cfg.epsilon(eps);
            return cfg;
        },
        "Tolerance for multiresolution adaptation (read/write, returns config for chaining).\n\n"
        "Cells with detail coefficients below this threshold may be coarsened.\n"
        "Default: 1e-4"
    );

    // Regularity property - mesh gradation parameter
    cls.def_property("regularity",
        [](const mra_config& cfg) {
            return cfg.regularity();
        },
        [](mra_config& cfg, double reg) {
            cfg.regularity(reg);
            return cfg;
        },
        "Regularity parameter controlling mesh gradation (read/write, returns config for chaining).\n\n"
        "Higher values enforce smoother transitions between refinement levels.\n"
        "Default: 1.0"
    );

    // Relative detail property - use relative vs absolute detail
    cls.def_property("relative_detail",
        [](const mra_config& cfg) {
            return cfg.relative_detail();
        },
        [](mra_config& cfg, bool rel) {
            cfg.relative_detail(rel);
            return cfg;
        },
        "Use relative detail criterion instead of absolute (read/write, returns config for chaining).\n\n"
        "When True, detail is normalized by maximum field values.\n"
        "Default: False"
    );
}

// Bind MRAConfig class to Python
void bind_mra_config(py::module_& m, const std::string& name)
{
    using namespace samurai;

    auto cls = py::class_<mra_config>(m, name.c_str(), R"pbdoc(
        Multiresolution Analysis (MRA) configuration for mesh adaptation.

        This class configures parameters for adaptive mesh refinement based on
        the Harten multiresolution analysis algorithm.

        Parameters
        ----------
        None - creates a configuration with default values

        Examples
        --------
        >>> import samurai_python as sam
        >>> config = sam.MRAConfig()
        >>> config.epsilon = 2e-4
        >>> config.regularity = 2.0
        >>> config.relative_detail = False

        Method chaining (fluent interface):

        >>> config = sam.MRAConfig().epsilon(2e-4).regularity(2.0)

        Usage with adaptation:

        >>> MRadaptation = samurai.make_MRAdapt(field)  # Returns Adapt object
        >>> MRadaptation(config)  # Apply adaptation with config

        Attributes
        ----------
        epsilon : float
            Tolerance for mesh adaptation (default: 1e-4).
            Cells with detail coefficients below this threshold may be coarsened.
        regularity : float
            Regularity parameter for mesh gradation (default: 1.0).
            Higher values enforce smoother transitions between refinement levels.
        relative_detail : bool
            Use relative detail criterion (default: False).
            When True, detail is normalized by maximum field values.

        Notes
        -----
        The same configuration object can be reused across multiple
        adaptation calls, for example in time loops.

        Typical epsilon values range from 1e-5 (very fine) to 1e-1 (coarse).
        Typical regularity values range from 0.0 (minimal gradation) to 3.0 (very smooth).
    )pbdoc");

    // Default constructor
    cls.def(py::init<>(),
        "Create MRA configuration with default values\n"
        "(epsilon=1e-4, regularity=1.0, relative_detail=False)"
    );

    // Bind properties
    bind_mra_config_properties(cls);

    // String representations
    cls.def("__repr__",
        [](const mra_config& cfg) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(4);
            oss << "MRAConfig(";
            oss << "epsilon=" << cfg.epsilon();
            oss << ", regularity=" << cfg.regularity();
            oss << ", relative_detail=" << (cfg.relative_detail() ? "True" : "False");
            oss << ")";
            return oss.str();
        },
        "Detailed string representation"
    );

    cls.def("__str__",
        [](const mra_config& cfg) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(4);
            oss << "MRAConfig";
            oss << " [epsilon=" << cfg.epsilon();
            oss << ", regularity=" << cfg.regularity();
            oss << ", relative_detail=" << (cfg.relative_detail() ? "True" : "False");
            oss << "]";
            return oss.str();
        },
        "Concise string representation"
    );

    // Equality operator (useful for testing)
    cls.def("__eq__",
        [](const mra_config& self, const mra_config& other) {
            return self.epsilon() == other.epsilon() &&
                   self.regularity() == other.regularity() &&
                   self.relative_detail() == other.relative_detail();
        },
        "Equality comparison"
    );
}

// Module initialization function for MRA configuration bindings
void init_mra_config_bindings(py::module_& m)
{
    bind_mra_config(m, "MRAConfig");
}
