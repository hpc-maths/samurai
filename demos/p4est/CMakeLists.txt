find_package(cxxopts)

add_executable(simple_2d simple_2d.cpp)# ${MURE_HEADERS})
target_link_libraries(simple_2d mure)
