enable_language (C Fortran)

set(RADAU_SRC
    decsol.f
    dc_decsol.f
    integration_stiff.c
    radau5.f
    rock4.f
)
add_library(radau ${RADAU_SRC})
