cmake_minimum_required(VERSION 3.10)
project(nucleon-qdq Fortran)

option(USE_MPI "build with MPI (mpif90, -DMPI)" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)
endif()

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/sources/*.f90")   #all f90 files are sources
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/debug")       #executable -> debug/
set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O2 -cpp -ffree-line-length-none")
#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O0 -g -fcheck=all -fbacktrace -cpp -ffree-line-length-none")  #debug
set(BLA_VENDOR OpenBLAS)                #multithreaded zgemm; remove to use the reference BLAS
find_package(LAPACK REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES})
if(USE_MPI)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MPI)
    target_link_libraries(${PROJECT_NAME} MPI::MPI_Fortran)
endif()
