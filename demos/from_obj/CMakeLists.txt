find_package(CGAL)

if(CGAL_FOUND)
    find_package(Eigen3 3.1.0 REQUIRED) # (3.1.0 or greater)
    include(CGAL_Eigen3_support)

    add_executable(mesh-from-obj main.cpp)

    target_compile_definitions(mesh-from-obj PUBLIC _HAS_AUTO_PTR_ETC=0)
    target_link_libraries(mesh-from-obj samurai CGAL::CGAL CGAL::Eigen3_support CLI11::CLI11)
endif()
