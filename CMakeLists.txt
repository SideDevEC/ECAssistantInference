cmake_minimum_required(VERSION 3.18)
project(ECAssistantInference VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
option(ECI_BACKEND_METAL  "Build with Metal backend (macOS)" OFF)
option(ECI_BACKEND_CUDA   "Build with CUDA backend" OFF)
option(ECI_BACKEND_VULKAN "Build with Vulkan backend" OFF)
option(ECI_BACKEND_CPU   "Build CPU-only backend" ON)

# ── llama.cpp submodule ──
# Expects llama.cpp as a subdirectory at ${CMAKE_SOURCE_DIR}/llama.cpp
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/llama.cpp/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp submodule not found. Run: git submodule update --init --recursive")
endif()

# Build llama.cpp as a static library (we link it into our shared lib)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)

if(ECI_BACKEND_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(ECI_BACKEND_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(ECI_BACKEND_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(llama.cpp)

# ── Our library ──
add_library(ecainference SHARED
    src/ecainference.cpp
    src/eci_backend.cpp
)

target_include_directories(ecainference
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/tools/mtmd
)

target_link_libraries(ecainference
    PRIVATE
        llama
        ggml
        ggml-base
        mtmd
)

target_compile_definitions(ecainference PRIVATE ECAINFERENCE_EXPORTS)

# ── Tests ──
enable_testing()

add_executable(ecainference_tests
    tests/test_basic.cpp
)

add_executable(ecainference_stress_tests
    tests/test_stress.cpp
)

target_link_libraries(ecainference_stress_tests
    PRIVATE
        ecainference
)

target_include_directories(ecainference_stress_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ecainference_tests
    PRIVATE
        ecainference
)

target_include_directories(ecainference_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME eci_tests COMMAND ecainference_tests)

add_test(NAME eci_stress_tests COMMAND ecainference_stress_tests)
