cmake_minimum_required(VERSION 3.21)

set(CMAKE_POLICY_DEFAULT_CMP0194 NEW)

project(qwen3_tts_cpp VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC AND CMAKE_CXX_COMPILER)
    set(
        CMAKE_ASM_COMPILER
        "${CMAKE_CXX_COMPILER}"
        CACHE FILEPATH
        "MSVC-compatible assembler compiler"
        FORCE
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 所有可执行文件与 ggml 的共享库统一输出到 <build>/bin/<config>，
# 这样 exe 旁边就是全部 backend DLL，不需要额外拷贝。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

set(
    QWEN3TTS_MODEL_DIR
    "${PROJECT_SOURCE_DIR}/models"
    CACHE PATH
    "Directory containing runtime GGUF models"
)
set(
    QWEN3TTS_TEST_DATA_DIR
    "${PROJECT_SOURCE_DIR}/testdata"
    CACHE PATH
    "Directory containing golden test fixtures"
)

set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries")

if(BUILD_SHARED_LIBS)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

option(
    QWEN3TTS_ENABLE_CUDA
    "Build Qwen3TTS with the GGML CUDA backend"
    OFF
)
set(
    QWEN3TTS_CUDA_ARCHITECTURES
    "native"
    CACHE STRING
    "CUDA architectures: native or a semicolon-separated list"
)
if(NOT QWEN3TTS_CUDA_ARCHITECTURES)
    set(
        QWEN3TTS_CUDA_ARCHITECTURES
        "native"
        CACHE STRING "" FORCE
    )
endif()

if(QWEN3TTS_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    if(QWEN3TTS_CUDA_ARCHITECTURES)
        set(
            CMAKE_CUDA_ARCHITECTURES
            "${QWEN3TTS_CUDA_ARCHITECTURES}"
            CACHE STRING "" FORCE
        )
    endif()
    message(STATUS "Qwen3TTS CUDA backend enabled")
    message(STATUS "Qwen3TTS CUDA architectures: "
        "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
    message(STATUS "Qwen3TTS CUDA backend disabled")
endif()

if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()

add_subdirectory(vendor/ggml-0.29.0)

function(qwen3_tts_configure_target target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/vendor
    )
    target_compile_definitions(${target} PRIVATE
        UNI_ALGO_STATIC_DATA
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
    endif()
    target_link_libraries(${target} PRIVATE ggml)
endfunction()

set(QWEN3_TTS_CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/config.cpp
    ${PROJECT_SOURCE_DIR}/src/gguf_model.cpp
    ${PROJECT_SOURCE_DIR}/src/talker.cpp
    ${PROJECT_SOURCE_DIR}/src/code_predictor.cpp
    ${PROJECT_SOURCE_DIR}/src/sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/text_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/text_tokenizer_internal.cpp
    ${PROJECT_SOURCE_DIR}/src/speech_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen3_tts.cpp
)

add_executable(qwen3tts-cli
    ${PROJECT_SOURCE_DIR}/app/qwen3tts_cli.cpp
    ${QWEN3_TTS_CORE_SOURCES}
    ${PROJECT_SOURCE_DIR}/src/wav_writer.cpp
)
qwen3_tts_configure_target(qwen3tts-cli)
target_compile_definitions(qwen3tts-cli PRIVATE
    QWEN3TTS_VERSION="${PROJECT_VERSION}"
)
if(WIN32)
    target_link_libraries(qwen3tts-cli PRIVATE shell32)
endif()

option(BUILD_TESTING "Build Qwen3TTS tests" OFF)
include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
