# add_subdirectory(RockAndRadau)

enable_language (Fortran)

set(RADAU_SRC
    RockAndRadau/decsol.f
    RockAndRadau/dc_decsol.f
    RockAndRadau/integration_stiff.cpp
    RockAndRadau/radau5.f
    RockAndRadau/rock4.f
)

add_executable(bz_2d bz_2d.cpp ${RADAU_SRC})
target_link_libraries(bz_2d samurai)

add_executable(bz_2d_AMR bz_2d_AMR.cpp ${RADAU_SRC})
target_link_libraries(bz_2d_AMR samurai)
