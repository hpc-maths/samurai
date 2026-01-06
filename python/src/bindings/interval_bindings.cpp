// Samurai Python Bindings - Interval class
//
// Bindings for samurai::Interval class
// Interval represents a half-open interval [start, end) with step and storage index

#include <samurai/interval.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <sstream>

namespace py = pybind11;

// Type aliases matching default Samurai configuration
// Note: Interval requires signed types for both value and index
using default_interval = samurai::Interval<int, long long int>;

template <class TValue, class TIndex>
void bind_interval_class(py::module_& m, const std::string& name) {
    using Interval = samurai::Interval<TValue, TIndex>;
    using value_t = typename Interval::value_t;
    using index_t = typename Interval::index_t;

    py::class_<Interval>(m, name.c_str(), R"pbdoc(
        Half-open interval [start, end) with step and storage index.

        An Interval represents a span of discrete coordinates in an adaptive mesh.
        It uses half-open interval semantics like Python ranges: [start, end).

        Parameters
        ----------
        start : int
            Interval start (inclusive)
        end : int
            Interval end (exclusive)
        index : int, optional
            Storage index offset (default: 0)

        Examples
        --------
        >>> import samurai as sam
        >>> i = sam.Interval(0, 10)
        >>> print(i)
        [0,10[@0:1
        >>> i.size
        10
        >>> i.contains(5)
        True
        >>> i.contains(10)
        False

        Attributes
        ----------
        start : int
            Interval start (inclusive)
        end : int
            Interval end (exclusive)
        step : int
            Step between coordinates (1 for all, 2 for even/odd)
        index : int
            Storage offset for array mapping
    )pbdoc")

        // Default constructor
        .def(py::init<>(), "Create empty interval [0, 0)")

        // Main constructor
        .def(py::init<value_t, value_t, index_t>(),
            py::arg("start"),
            py::arg("end"),
            py::arg("index") = 0,
            "Create interval [start, end) with optional storage index")

        // Properties (read-write for all except step which is modified by operations)
        .def_property("start",
            [](const Interval& i) { return i.start; },
            [](Interval& i, value_t v) { i.start = v; },
            "Interval start (inclusive)"
        )

        .def_property("end",
            [](const Interval& i) { return i.end; },
            [](Interval& i, value_t v) { i.end = v; },
            "Interval end (exclusive)"
        )

        .def_property("step",
            [](const Interval& i) { return i.step; },
            [](Interval& i, value_t v) { i.step = v; },
            "Step between coordinates (1=continuous, 2=even/odd)"
        )

        .def_property("index",
            [](const Interval& i) { return i.index; },
            [](Interval& i, index_t v) { i.index = v; },
            "Storage index offset"
        )

        // Query methods
        .def("size", &Interval::size,
            "Number of elements in interval (end - start)"
        )

        .def("contains", &Interval::contains,
            py::arg("x"),
            "Check if coordinate x is within [start, end)"
        )

        .def("is_valid", &Interval::is_valid,
            "Check if interval is non-empty (start < end)"
        )

        .def("is_empty", &Interval::is_empty,
            "Check if interval is empty (start == end)"
        )

        // Element selection
        .def("even_elements", &Interval::even_elements,
            "Extract even-indexed elements (step becomes 2)"
        )

        .def("odd_elements", &Interval::odd_elements,
            "Extract odd-indexed elements (step becomes 2)"
        )

        // Compound assignment operators (in-place)
        .def("__imul__", [](Interval& i, value_t v) -> Interval& { return i *= v; },
            py::arg("v"),
            "Scale interval in-place: start *= v, end *= v"
        )

        .def("__itruediv__", [](Interval& i, value_t v) -> Interval& { return i /= v; },
            py::arg("v"),
            "Divide interval in-place with flooring"
        )

        .def("__irshift__", [](Interval& i, long long int v) -> Interval& { return i >>= v; },
            py::arg("v"),
            "Coarsen interval: divide coordinates by 2^v (decrease level)"
        )

        .def("__ilshift__", [](Interval& i, long long int v) -> Interval& { return i <<= v; },
            py::arg("v"),
            "Refine interval: multiply coordinates by 2^v (increase level)"
        )

        .def("__iadd__", [](Interval& i, value_t v) -> Interval& { return i += v; },
            py::arg("v"),
            "Shift interval right by v"
        )

        .def("__isub__", [](Interval& i, value_t v) -> Interval& { return i -= v; },
            py::arg("v"),
            "Shift interval left by v"
        )

        // Binary operators (return new Interval)
        .def("__mul__", [](const Interval& i, value_t v) { return i * v; },
            py::arg("v"),
            "Scale interval: return new interval with start*=-v, end*=v"
        )

        .def("__rmul__", [](const Interval& i, value_t v) { return v * i; },
            py::arg("v"),
            "Scale interval (right multiply)"
        )

        .def("__truediv__", [](const Interval& i, value_t v) { return i / v; },
            py::arg("v"),
            "Divide interval with flooring"
        )

        .def("__rshift__", [](const Interval& i, long long int v) { return i >> v; },
            py::arg("v"),
            "Coarsen interval: return new interval at lower level"
        )

        .def("__lshift__", [](const Interval& i, long long int v) { return i << v; },
            py::arg("v"),
            "Refine interval: return new interval at higher level"
        )

        .def("__add__", [](const Interval& i, value_t v) { return i + v; },
            py::arg("v"),
            "Shift interval right by v"
        )

        .def("__radd__", [](const Interval& i, value_t v) { return v + i; },
            py::arg("v"),
            "Shift interval right (right add)"
        )

        .def("__sub__", [](const Interval& i, value_t v) { return i - v; },
            py::arg("v"),
            "Shift interval left by v"
        )

        .def("__rsub__", [](const Interval& i, value_t v) { return v - i; },
            py::arg("v"),
            "Negate and shift (right subtract)"
        )

        // Comparison operators
        .def(py::self == py::self, "Full equality (start, end, step, index)")
        .def(py::self != py::self, "Inequality")
        .def(py::self < py::self, "Compare by start coordinate only")

        // String representations
        .def("__repr__", [](const Interval& i) {
            std::ostringstream oss;
            oss << "Interval(start=" << i.start
                << ", end=" << i.end
                << ", index=" << i.index
                << ", step=" << i.step << ")";
            return oss.str();
        })

        .def("__str__", [](const Interval& i) {
            std::ostringstream oss;
            oss << i;  // Uses C++ stream operator: "[start,end[@index:step"
            return oss.str();
        })

        // Length protocol (len(interval))
        .def("__len__", &Interval::size,
            "Number of elements in interval"
        )

        // Containment protocol (x in interval)
        .def("__contains__", &Interval::contains,
            "Check if value is in interval"
        );
}

// Module initialization function for Interval bindings
void init_interval_bindings(py::module_& m) {
    // Bind default Interval type (int coordinates, signed long long int index)
    bind_interval_class<int, long long int>(m, "Interval");

    // Factory function for convenience
    m.def("make_interval",
        [](int start, int end, long long int index = 0) {
            return default_interval(start, end, index);
        },
        py::arg("start"),
        py::arg("end"),
        py::arg("index") = 0,
        "Create an Interval [start, end) with optional storage index"
    );

    // Also expose in interval submodule for better organization
    py::module_ interval = m.def_submodule("interval", "Interval class");
    interval.attr("Interval") = m.attr("Interval");
}
