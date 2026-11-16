set(QWEN3_TTS_TEST_GGUF
    "${PROJECT_SOURCE_DIR}/testdata/qwen3-tts-talker-0.6b-customvoice-bf16.gguf"
)
set(QWEN3_TTS_TEST_SPEECH_GGUF
    "${PROJECT_SOURCE_DIR}/testdata/qwen3-tts-speech-tokenizer-12hz-f32.gguf"
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
        "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
    )
    qwen3_tts_configure_target(test_text_tokenizer)
    target_compile_definitions(test_text_tokenizer PRIVATE
        QWEN3_TTS_TEST_GGUF="${QWEN3_TTS_TEST_GGUF}"
    )
    add_test(NAME text_tokenizer COMMAND test_text_tokenizer)

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
            "${PROJECT_SOURCE_DIR}/testdata/qwen3-tts-talker-1.7b-customvoice-bf16.gguf"
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
