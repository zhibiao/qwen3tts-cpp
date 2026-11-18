set(QWEN3_TTS_TEST_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-talker-0.6b-customvoice-bf16.gguf"
)
set(QWEN3_TTS_TEST_SPEECH_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-speech-tokenizer-12hz-f32.gguf"
)

if(EXISTS "${QWEN3_TTS_TEST_GGUF}" AND
   EXISTS "${QWEN3_TTS_TEST_SPEECH_GGUF}")
    add_executable(test_gguf_model
        "${CMAKE_CURRENT_SOURCE_DIR}/test_gguf_model.cpp"
        "${PROJECT_SOURCE_DIR}/src/config.cpp"
        "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
    )
    qwen3_tts_configure_target(test_gguf_model)
    target_compile_definitions(test_gguf_model PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
    )
    add_test(NAME gguf_model COMMAND test_gguf_model)

    add_executable(test_qwen3tts_context
        "${CMAKE_CURRENT_SOURCE_DIR}/test_qwen3tts_context.cpp"
        ${QWEN3_TTS_CORE_SOURCES}
    )
    qwen3_tts_configure_target(test_qwen3tts_context)
    target_compile_definitions(test_qwen3tts_context PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
    )
    add_test(NAME qwen3tts_context COMMAND test_qwen3tts_context)

    add_executable(test_wav_writer
        "${CMAKE_CURRENT_SOURCE_DIR}/test_wav_writer.cpp"
        "${PROJECT_SOURCE_DIR}/src/wav_writer.cpp"
    )
    qwen3_tts_configure_target(test_wav_writer)
    add_test(NAME wav_writer COMMAND test_wav_writer)

    add_executable(test_speech_transposed_conv
        "${CMAKE_CURRENT_SOURCE_DIR}/test_speech_transposed_conv.cpp"
    )
    qwen3_tts_configure_target(test_speech_transposed_conv)
    add_test(
        NAME speech_transposed_conv
        COMMAND test_speech_transposed_conv
    )

    add_executable(test_sampler
        "${CMAKE_CURRENT_SOURCE_DIR}/test_sampler.cpp"
        "${PROJECT_SOURCE_DIR}/src/sampler.cpp"
    )
    qwen3_tts_configure_target(test_sampler)
    add_test(NAME sampler COMMAND test_sampler)

    add_executable(test_text_tokenizer
        "${CMAKE_CURRENT_SOURCE_DIR}/test_text_tokenizer.cpp"
        "${PROJECT_SOURCE_DIR}/src/config.cpp"
        "${PROJECT_SOURCE_DIR}/src/text_tokenizer.cpp"
        "${PROJECT_SOURCE_DIR}/src/text_tokenizer_internal.cpp"
        "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
    )
    qwen3_tts_configure_target(test_text_tokenizer)
    target_compile_definitions(test_text_tokenizer PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
    )
    add_test(NAME text_tokenizer COMMAND test_text_tokenizer)

    add_executable(test_text_tokenizer_unicode
        "${CMAKE_CURRENT_SOURCE_DIR}/test_text_tokenizer_unicode.cpp"
        "${PROJECT_SOURCE_DIR}/src/text_tokenizer_internal.cpp"
    )
    qwen3_tts_configure_target(test_text_tokenizer_unicode)
    add_test(
        NAME text_tokenizer_unicode
        COMMAND test_text_tokenizer_unicode
    )

    add_executable(test_speech_tokenizer
        "${CMAKE_CURRENT_SOURCE_DIR}/test_speech_tokenizer.cpp"
        ${QWEN3_TTS_CORE_SOURCES}
    )
    qwen3_tts_configure_target(test_speech_tokenizer)
    target_compile_definitions(test_speech_tokenizer PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
    )
    add_test(NAME speech_tokenizer COMMAND test_speech_tokenizer)

    add_executable(test_talker
        "${CMAKE_CURRENT_SOURCE_DIR}/test_talker.cpp"
        ${QWEN3_TTS_CORE_SOURCES}
    )
    qwen3_tts_configure_target(test_talker)
    target_compile_definitions(test_talker PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
    )
    add_test(NAME talker COMMAND test_talker)

    add_executable(test_code_predictor
        "${CMAKE_CURRENT_SOURCE_DIR}/test_code_predictor.cpp"
        ${QWEN3_TTS_CORE_SOURCES}
    )
    qwen3_tts_configure_target(test_code_predictor)
    target_compile_definitions(test_code_predictor PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
    )
    add_test(NAME code_predictor COMMAND test_code_predictor)

    set(QWEN3_TTS_TEST_GGUF_F32
        "${QWEN3_TTS_TEST_GGUF}"
    )
    set(QWEN3_TTS_TALKER_GOLDEN_DIR
        "${PROJECT_SOURCE_DIR}/testdata/talker-golden"
    )
    if(EXISTS "${QWEN3_TTS_TEST_GGUF_F32}" AND
       EXISTS "${QWEN3_TTS_TALKER_GOLDEN_DIR}/manifest.json")
        add_executable(test_talker_golden
            "${CMAKE_CURRENT_SOURCE_DIR}/test_talker_golden.cpp"
            ${QWEN3_TTS_CORE_SOURCES}
        )
        qwen3_tts_configure_target(test_talker_golden)
        target_compile_definitions(test_talker_golden PRIVATE
            QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_F32}"
            QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
            QWEN3_TTS_TALKER_GOLDEN_DIR="${QWEN3_TTS_TALKER_GOLDEN_DIR}"
        )
        add_test(NAME talker_golden COMMAND test_talker_golden)

        set(QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR
            "${PROJECT_SOURCE_DIR}/testdata/code-predictor-golden"
        )
        if(EXISTS "${QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR}/manifest.json")
            add_executable(test_code_predictor_golden
                "${CMAKE_CURRENT_SOURCE_DIR}/test_code_predictor_golden.cpp"
                ${QWEN3_TTS_CORE_SOURCES}
            )
            qwen3_tts_configure_target(test_code_predictor_golden)
            target_compile_definitions(test_code_predictor_golden PRIVATE
                QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_F32}"
                QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
                QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR="${QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR}"
            )
            add_test(NAME code_predictor_golden COMMAND test_code_predictor_golden)
        else()
            message(STATUS
                "CodePredictor golden test disabled; generate fixtures first"
            )
        endif()

        set(QWEN3_TTS_STREAM_GOLDEN_DIR
            "${PROJECT_SOURCE_DIR}/testdata/qwen3tts-stream-golden"
        )
        if(EXISTS "${QWEN3_TTS_STREAM_GOLDEN_DIR}/manifest.json")
            add_executable(test_qwen3tts_golden
                "${CMAKE_CURRENT_SOURCE_DIR}/test_qwen3tts_golden.cpp"
                ${QWEN3_TTS_CORE_SOURCES}
            )
            qwen3_tts_configure_target(test_qwen3tts_golden)
            target_compile_definitions(test_qwen3tts_golden PRIVATE
                QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_F32}"
                QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
                QWEN3_TTS_STREAM_GOLDEN_DIR="${QWEN3_TTS_STREAM_GOLDEN_DIR}"
            )
            add_test(NAME qwen3tts_golden COMMAND test_qwen3tts_golden)
        else()
            message(STATUS
                "Qwen3TTS multi-frame golden test disabled; generate fixtures first"
            )
        endif()

        set(QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR
            "${PROJECT_SOURCE_DIR}/testdata/custom-voice-golden"
        )
        if(EXISTS "${QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR}/manifest.json")
            add_executable(test_qwen3tts_customvoice_golden
                "${CMAKE_CURRENT_SOURCE_DIR}/test_qwen3tts_customvoice_golden.cpp"
                ${QWEN3_TTS_CORE_SOURCES}
            )
            qwen3_tts_configure_target(test_qwen3tts_customvoice_golden)
            target_compile_definitions(test_qwen3tts_customvoice_golden PRIVATE
                QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_F32}"
                QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
                QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR="${QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR}"
            )
            add_test(
                NAME qwen3tts_customvoice_golden
                COMMAND test_qwen3tts_customvoice_golden
            )
            set_tests_properties(
                qwen3tts_customvoice_golden
                PROPERTIES DISABLED TRUE
            )
        else()
            message(STATUS
                "CustomVoice golden test disabled; generate fixtures first"
            )
        endif()

        set(QWEN3_TTS_TEST_GGUF_1_7_F32
            "${PROJECT_SOURCE_DIR}/models/qwen3-tts-talker-1.7b-customvoice-bf16.gguf"
        )
        set(QWEN3_TTS_1_7_GOLDEN_DIR
            "${PROJECT_SOURCE_DIR}/testdata"
        )
        if(EXISTS "${QWEN3_TTS_TEST_GGUF_1_7_F32}" AND
           EXISTS "${QWEN3_TTS_1_7_GOLDEN_DIR}/1.7b-reference-3-codes.bin")
            add_executable(test_code_predictor_1_7_golden
                "${CMAKE_CURRENT_SOURCE_DIR}/test_code_predictor_1_7_golden.cpp"
                ${QWEN3_TTS_CORE_SOURCES}
            )
            qwen3_tts_configure_target(test_code_predictor_1_7_golden)
            target_compile_definitions(test_code_predictor_1_7_golden PRIVATE
                QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_1_7_F32}"
                QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
                QWEN3_TTS_1_7_GOLDEN_DIR="${QWEN3_TTS_1_7_GOLDEN_DIR}"
            )
            add_test(
                NAME code_predictor_1_7_golden
                COMMAND test_code_predictor_1_7_golden
            )

            if(EXISTS
               "${QWEN3_TTS_1_7_GOLDEN_DIR}/1.7b-reference-prefill.bin")
                add_executable(test_talker_1_7_golden
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_talker_1_7_golden.cpp"
                    ${QWEN3_TTS_CORE_SOURCES}
                )
                qwen3_tts_configure_target(test_talker_1_7_golden)
                target_compile_definitions(test_talker_1_7_golden PRIVATE
                    QWEN3_TTS_TEST_GGUF_F32="${QWEN3_TTS_TEST_GGUF_1_7_F32}"
                    QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_TEST_SPEECH_GGUF}"
                    QWEN3_TTS_1_7_GOLDEN_DIR="${QWEN3_TTS_1_7_GOLDEN_DIR}"
                )
                add_test(
                    NAME talker_1_7_golden
                    COMMAND test_talker_1_7_golden
                )
            endif()
        endif()
    else()
        message(STATUS
            "Talker golden test disabled; generate F32 GGUF and fixtures first"
        )
    endif()
else()
    message(WARNING
        "Full test GGUF not found; run convert_hf_to_gguf.py to enable integration tests"
    )
endif()

set(QWEN3_TTS_EXTRA_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-talker-0.6b-customvoice-bf16.gguf"
)
set(QWEN3_TTS_EXTRA_SPEECH_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-speech-tokenizer-12hz-f32.gguf"
)
set(QWEN3_TTS_EXTRA_Q8_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-talker-0.6b-customvoice-q8_0.gguf"
)
set(QWEN3_TTS_EXTRA_1_7_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-tts-talker-1.7b-customvoice-bf16.gguf"
)
set(QWEN3_TTS_EXTRA_TEXT_GOLDEN
    "${PROJECT_SOURCE_DIR}/testdata/text-tokenizer-golden/cases.json"
)
set(QWEN3_TTS_EXTRA_CUSTOMVOICE_GOLDEN
    "${PROJECT_SOURCE_DIR}/testdata/custom-voice-golden"
)
set(QWEN3_TTS_EXTRA_E2E_GOLDEN
    "${PROJECT_SOURCE_DIR}/testdata/qwen3tts-e2e-golden"
)

if(NOT EXISTS "${QWEN3_TTS_EXTRA_GGUF}" OR
   NOT EXISTS "${QWEN3_TTS_EXTRA_SPEECH_GGUF}")
    message(STATUS "Extended integration tests disabled; model GGUF files missing")
    return()
endif()

function(add_extra_test target source)
    add_executable(${target}
        "${PROJECT_SOURCE_DIR}/tests/${source}"
        ${ARGN}
    )
    qwen3_tts_configure_target(${target})
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/tests"
    )
    target_compile_definitions(${target} PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_EXTRA_GGUF}"
        QWEN3_TTS_TEST_SPEECH_GGUF="${QWEN3_TTS_EXTRA_SPEECH_GGUF}"
    )
endfunction()

if(EXISTS "${QWEN3_TTS_EXTRA_TEXT_GOLDEN}")
    add_extra_test(test_text_tokenizer_scenarios
        test_text_tokenizer_scenarios.cpp
        "${PROJECT_SOURCE_DIR}/src/config.cpp"
        "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
        "${PROJECT_SOURCE_DIR}/src/text_tokenizer.cpp"
        "${PROJECT_SOURCE_DIR}/src/text_tokenizer_internal.cpp"
    )
    target_include_directories(test_text_tokenizer_scenarios PRIVATE
        "${PROJECT_SOURCE_DIR}/vendor"
    )
    target_compile_definitions(test_text_tokenizer_scenarios PRIVATE
        QWEN3_TTS_TEXT_GOLDEN="${QWEN3_TTS_EXTRA_TEXT_GOLDEN}"
    )
    add_test(NAME text_tokenizer_scenarios
        COMMAND test_text_tokenizer_scenarios)
endif()

if(EXISTS "${QWEN3_TTS_EXTRA_CUSTOMVOICE_GOLDEN}/manifest.json")
    add_extra_test(test_speech_tokenizer_golden
        test_speech_tokenizer_golden.cpp
        ${QWEN3_TTS_CORE_SOURCES}
    )
    target_compile_definitions(test_speech_tokenizer_golden PRIVATE
        QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR="${QWEN3_TTS_EXTRA_CUSTOMVOICE_GOLDEN}"
    )
    add_test(NAME speech_tokenizer_golden
        COMMAND test_speech_tokenizer_golden)

    add_extra_test(test_custom_voice_prompt_golden
        test_custom_voice_prompt_golden.cpp
        ${QWEN3_TTS_CORE_SOURCES}
    )
    target_compile_definitions(test_custom_voice_prompt_golden PRIVATE
        QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR="${QWEN3_TTS_EXTRA_CUSTOMVOICE_GOLDEN}"
    )
    add_test(NAME custom_voice_prompt_golden
        COMMAND test_custom_voice_prompt_golden)
endif()

if(EXISTS "${QWEN3_TTS_EXTRA_E2E_GOLDEN}/manifest.json")
    add_extra_test(test_qwen3tts_e2e_golden
        test_qwen3tts_e2e_golden.cpp
        ${QWEN3_TTS_CORE_SOURCES}
    )
    target_include_directories(test_qwen3tts_e2e_golden PRIVATE
        "${PROJECT_SOURCE_DIR}/vendor"
    )
    target_compile_definitions(test_qwen3tts_e2e_golden PRIVATE
        QWEN3_TTS_E2E_GOLDEN_DIR="${QWEN3_TTS_EXTRA_E2E_GOLDEN}"
        QWEN3_TTS_E2E_GOLDEN_MANIFEST="${QWEN3_TTS_EXTRA_E2E_GOLDEN}/manifest.json"
    )
    if(EXISTS "${QWEN3_TTS_EXTRA_1_7_GGUF}")
        target_compile_definitions(test_qwen3tts_e2e_golden PRIVATE
            QWEN3_TTS_TEST_GGUF_1_7="${QWEN3_TTS_EXTRA_1_7_GGUF}"
        )
    endif()
    add_test(NAME qwen3tts_e2e_golden
        COMMAND test_qwen3tts_e2e_golden)
endif()

if(QWEN3TTS_ENABLE_CUDA AND
   EXISTS "${PROJECT_SOURCE_DIR}/testdata/talker-golden/manifest.json" AND
   EXISTS "${PROJECT_SOURCE_DIR}/testdata/code-predictor-golden/manifest.json")
    add_extra_test(test_cuda_golden
        test_cuda_golden.cpp
        ${QWEN3_TTS_CORE_SOURCES}
    )
    target_compile_definitions(test_cuda_golden PRIVATE
        QWEN3_TTS_TALKER_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/testdata/talker-golden"
        QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/testdata/code-predictor-golden"
    )
    add_test(NAME cuda_golden COMMAND test_cuda_golden)
    set_tests_properties(cuda_golden PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_extra_test(test_sampler_extended
    test_sampler_extended.cpp
    "${PROJECT_SOURCE_DIR}/src/sampler.cpp"
)
add_test(NAME sampler_extended COMMAND test_sampler_extended)

add_extra_test(test_wav_writer_extended
    test_wav_writer_extended.cpp
    "${PROJECT_SOURCE_DIR}/src/wav_writer.cpp"
)
add_test(NAME wav_writer_extended COMMAND test_wav_writer_extended)

add_extra_test(test_gguf_config_validation
    test_gguf_config_validation.cpp
    "${PROJECT_SOURCE_DIR}/src/config.cpp"
    "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
)
if(EXISTS "${QWEN3_TTS_EXTRA_Q8_GGUF}")
    target_compile_definitions(test_gguf_config_validation PRIVATE
        QWEN3_TTS_TEST_GGUF_Q8="${QWEN3_TTS_EXTRA_Q8_GGUF}"
    )
endif()
add_test(NAME gguf_config_validation COMMAND test_gguf_config_validation)

add_extra_test(test_context_extended
    test_context_extended.cpp
    ${QWEN3_TTS_CORE_SOURCES}
)
if(EXISTS "${QWEN3_TTS_EXTRA_Q8_GGUF}")
    target_compile_definitions(test_context_extended PRIVATE
        QWEN3_TTS_TEST_GGUF_Q8="${QWEN3_TTS_EXTRA_Q8_GGUF}"
    )
endif()
add_test(NAME context_extended COMMAND test_context_extended)

add_extra_test(test_cli_wav test_cli_wav.cpp)

function(add_cli_case name text)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/cli-${name}.wav")
    add_test(NAME "${name}"
        COMMAND qwen3tts-cli
            --model "${QWEN3_TTS_EXTRA_GGUF}"
            --speech-tokenizer "${QWEN3_TTS_EXTRA_SPEECH_GGUF}"
            --text "${text}"
            --speaker serena
            --language Chinese
            --max-new-tokens 2
            --seed 1234
            --greedy
            --device CPU
            --threads 2
            --output "${output}"
    )
    set_tests_properties("${name}" PROPERTIES
        FIXTURES_SETUP "${name}"
    )
    add_test(NAME "${name}_wav"
        COMMAND test_cli_wav "${output}")
    set_tests_properties("${name}_wav" PROPERTIES
        FIXTURES_REQUIRED "${name}"
    )
endfunction()

add_cli_case(cli_short_zh "你好。")
add_cli_case(cli_punctuation_zh "你好，世界！今天天气怎么样？")
add_cli_case(cli_english "Hello, world! How are you?")
add_cli_case(cli_mixed "你好，OpenAI。今天讨论 Qwen3-TTS（12Hz）与 C++/ggml，OK?")
add_cli_case(cli_long_mixed
    "今天的测试包含较长的中文句子，也包含 English words、数字 12345、标点符号：逗号，句号。问号？感叹号！以及括号（测试）和斜杠 A/B。我们继续增加上下文，以覆盖不同长度的输入。"
)

if(EXISTS "${QWEN3_TTS_EXTRA_1_7_GGUF}")
    set(output_1_7 "${CMAKE_CURRENT_BINARY_DIR}/cli-1.7-short.wav")
    add_test(NAME cli_1_7_short
        COMMAND qwen3tts-cli
            --model "${QWEN3_TTS_EXTRA_1_7_GGUF}"
            --speech-tokenizer "${QWEN3_TTS_EXTRA_SPEECH_GGUF}"
            --text "你好，Qwen3-TTS。"
            --speaker serena
            --language Chinese
            --max-new-tokens 2
            --seed 1234
            --greedy
            --device CPU
            --threads 2
            --output "${output_1_7}"
    )
    set_tests_properties(cli_1_7_short PROPERTIES
        FIXTURES_SETUP cli_1_7_short
    )
    add_test(NAME cli_1_7_short_wav
        COMMAND test_cli_wav "${output_1_7}"
    )
    set_tests_properties(cli_1_7_short_wav PROPERTIES
        FIXTURES_REQUIRED cli_1_7_short
    )
endif()

set(QWEN3_TTS_EXTRA_PYTHON
    "${PROJECT_SOURCE_DIR}/.venv/Scripts/python.exe")
if(EXISTS "${QWEN3_TTS_EXTRA_PYTHON}")
    add_test(NAME quantize_gguf_presets
        COMMAND "${QWEN3_TTS_EXTRA_PYTHON}"
            "${PROJECT_SOURCE_DIR}/tests/test_quantize_gguf.py"
    )
    set_tests_properties(quantize_gguf_presets PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
    add_test(NAME convert_hf_helpers
        COMMAND "${QWEN3_TTS_EXTRA_PYTHON}"
            "${PROJECT_SOURCE_DIR}/tests/test_convert_hf_to_gguf.py"
    )
    set_tests_properties(convert_hf_helpers PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
endif()
