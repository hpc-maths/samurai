from conans import ConanFile, CMake

class SamuraiConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = [
        "xtensor/0.24.2",
        "highfive/2.6.2",
        "pugixml/1.13",
        "cli11/2.3.2",
        "cxxopts/3.0.0",
        "fmt/9.1.0",
        "rapidcheck/cci.20220514",
    ]
    generators = ["CMakeDeps", "CMakeToolchain"]
    default_options = {
        "hdf5:shared": True,
        "highfive:with_boost": False,
        "highfive:with_opencv": False,
        "highfive:with_eigen": False,
        "rapidcheck:enable_gtest": True,
    }

    def configure(self):
        if self.settings.os == "Windows" and self.settings.compiler == "msvc":
            self.options["hdf5:shared"] = True

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()