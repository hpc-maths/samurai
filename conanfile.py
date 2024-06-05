from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class SamuraiConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "build_demos": [True, False],
        "build_tests": [True, False],
    }
    default_options = {
        "build_demos": False,
        "build_tests": False,
    }
    generators = ["CMakeDeps", "CMakeToolchain"]
    default_options = {
        "hdf5/*:shared": False,
        "highfive/*:with_boost": False,
        "highfive/*:with_opencv": False,
        "highfive/*:with_eigen": False,
        "rapidcheck/*:enable_gtest": True,
    }

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("xtensor/0.24.7")
        self.requires("highfive/2.9.0")
        self.requires("hdf5/1.14.3")
        self.requires("pugixml/1.14")
        self.requires("cli11/2.4.2")
        self.requires("cxxopts/3.2.0")
        self.requires("fmt/10.2.1")
        self.requires("rapidcheck/cci.20230815")
        if self.options.build_demos:
            self.requires("cgal/5.6.1")
        if self.options.build_tests:
            self.requires("gtest/1.14.0")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
