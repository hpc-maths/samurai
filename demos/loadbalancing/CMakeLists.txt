include(FindPkgConfig)

# add_executable(test-simple-2d partition.cpp)
# target_link_libraries(test-simple-2d samurai CLI11::CLI11)

add_executable(test-circle-2d circle.cpp)
target_link_libraries(test-circle-2d samurai CLI11::CLI11 ${MPI_LIBRARIES})