from conan.tools.files import copy
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, cmake_layout
import os


class CppRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    default_options = {
        "spdlog/*:use_std_fmt": True,
        "glad/*:gl_profile": "core",
        "glad/*:gl_version": "4.6"
    }

    def requirements(self):
        self.requires("spdlog/1.15.1")
        self.requires("eigen/3.4.0")
        # self.requires("zeromq/4.3.5")
        # self.requires("cppzmq/4.10.0")
        # self.requires("nlohmann_json/3.12.0")
        self.requires("catch2/3.8.0")
        self.requires("imgui/1.91.0-docking", override=True)
        self.requires("implot/0.16")
        self.requires("glad/0.1.36")
        self.requires("glfw/3.4")
        # self.requires("doctest/2.4.11")
        # self.requires("boost/1.87.0")
        # self.requires("raylib/5.5")
        # self.requires("glm/1.0.1")

    # def generate(self):
        # Copy the source code and some of the extra src files to the build/imgui_src folder
        # copy(self, "res/*", self.dependencies["imgui"].package_folder,
        #     os.path.join(self.build_folder, "imgui_src") )
        # copy(self, "include/*", self.dependencies["imgui"].package_folder,
        #     os.path.join(self.build_folder, "imgui_src") )
