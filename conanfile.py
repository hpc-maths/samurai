from conan import ConanFile, CMake

class SamuraiConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = [
        "xtensor/0.24.6",
        "highfive/2.7.1",
        "hdf5/1.14.3",
        "pugixml/1.14",
        "cli11/3.2.0",
        "cxxopts/3.0.0",
        "fmt/10.1.1",
        "rapidcheck/cci.20220514",
    ]
    generators = ["CMakeDeps", "CMakeToolchain"]
    default_options = {
        "hdf5:shared": False,
        "highfive:with_boost": False,
        "highfive:with_opencv": False,
        "highfive:with_eigen": False,
        "rapidcheck:enable_gtest": True,
    }

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
