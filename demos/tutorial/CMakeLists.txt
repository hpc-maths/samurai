add_subdirectory(AMR_1D_Burgers)

add_executable(tutorial-2d-mesh 2D_mesh.cpp)
target_link_libraries(tutorial-2d-mesh PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-interval interval.cpp)
target_link_libraries(tutorial-interval PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-set-operator set_operator.cpp)
target_link_libraries(tutorial-set-operator PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-graduation-case-1 graduation_case_1.cpp)
target_link_libraries(tutorial-graduation-case-1 PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-graduation-case-2 graduation_case_2.cpp)
target_link_libraries(tutorial-graduation-case-2 PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-graduation-case-3 graduation_case_3.cpp)
target_link_libraries(tutorial-graduation-case-3 PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-reconstruction-1d reconstruction_1d.cpp)
target_link_libraries(tutorial-reconstruction-1d PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-reconstruction-2d reconstruction_2d.cpp)
target_link_libraries(tutorial-reconstruction-2d PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-reconstruction-3d reconstruction_3d.cpp)
target_link_libraries(tutorial-reconstruction-3d PRIVATE samurai CLI11::CLI11)

add_executable(tutorial-proj-on-mesh proj_on_mesh.cpp)
target_link_libraries(tutorial-proj-on-mesh PRIVATE samurai CLI11::CLI11)

if(MSVC)
    target_compile_options(tutorial-reconstruction-3d PUBLIC /bigobj)
endif()
