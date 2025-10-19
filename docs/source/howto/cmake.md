# How-to: create a CMake project for samurai

This how-to guide will show you how to create a simple CMake project that uses samurai.

Before creating a CMake project, make sure you have samurai installed. If you haven't installed it yet, please refer to the installation guide [here](installation.md).

## Create a CMake project

To create a CMake project that uses samurai, you need to create a `CMakeLists.txt` file in your project directory. Here is a simple example of a `CMakeLists.txt` file that includes samurai:

```{literalinclude} snippet/cmake/CMakeLists.txt
  :language: cmake
```

In this example, we first specify the minimum required version of CMake and the project name. Then, we use the `find_package` command to locate the samurai library. Finally, we create an executable target called `my_executable` and link it with the samurai library.

## Build the CMake project

Let's create a simple `main.cpp` file to test our CMake project:

```{literalinclude} snippet/cmake/main.cpp
  :language: c++
```

To build the CMake project, you can use the following commands:

```bash
cmake -S . -B build
cmake --build build
```

This will generate the build files in the `build` directory and then compile the project.

And that's it! You have successfully created a CMake project that uses samurai.