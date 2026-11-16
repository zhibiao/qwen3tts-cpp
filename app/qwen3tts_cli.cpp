#include "qwen3_tts.h"
#include "wav_writer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

struct Arguments {
    std::string talker_model;
    std::string speech_model;
    std::string text;
    std::string speaker = "serena";
    std::string language = "Chinese";
    std::string instruct;
    std::string output = "qwen3tts-cli.wav";
    std::string codes_output;
    std::string device;
    int32_t threads = 0;
    std::optional<int32_t> max_new_tokens;
    bool greedy = false;
    bool talker_greedy = false;
    bool subtalker_greedy = false;
    std::optional<float> repetition_penalty;
    uint64_t seed = 0;
    bool list_devices = false;
};

bool parse_args(int argc, char** argv, Arguments& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--list-devices") {
            args.list_devices = true;
            continue;
        }
        if (option == "--greedy") {
            args.greedy = true;
            continue;
        }
        if (option == "--talker-greedy") {
            args.talker_greedy = true;
            continue;
        }
        if (option == "--subtalker-greedy") {
            args.subtalker_greedy = true;
            continue;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const std::string value = argv[++i];
        if (option == "--model") {
            args.talker_model = value;
        } else if (
            option == "--speech-tokenizer" ||
            option == "--codec") {
            args.speech_model = value;
        } else if (option == "--text") {
            args.text = value;
        } else if (option == "--speaker") {
            args.speaker = value;
        } else if (option == "--language") {
            args.language = value;
        } else if (option == "--instruct") {
            args.instruct = value;
        } else if (option == "--output") {
            args.output = value;
        } else if (option == "--codes-output") {
            args.codes_output = value;
        } else if (option == "--device") {
            args.device = value;
        } else if (option == "--threads") {
            const int32_t parsed = std::atoi(value.c_str());
            if (parsed < 0) {
                return false;
            }
            args.threads = parsed;
        } else if (option == "--max-new-tokens") {
            const int32_t parsed = std::atoi(value.c_str());
            if (parsed <= 0) {
                return false;
            }
            args.max_new_tokens = parsed;
        } else if (option == "--seed") {
            args.seed = std::strtoull(value.c_str(), nullptr, 10);
        } else if (option == "--repetition-penalty") {
            args.repetition_penalty = std::strtof(value.c_str(), nullptr);
        } else {
            return false;
        }
    }
    return args.list_devices ||
        (!args.talker_model.empty() &&
         !args.speech_model.empty() &&
         !args.text.empty());
}

void usage() {
    std::cerr
        << "Usage: qwen3tts-cli --model TALKER.gguf "
           "--speech-tokenizer SPEECH.gguf --text TEXT "
           "[--speaker NAME] [--language NAME] [--instruct TEXT] "
           "[--max-new-tokens N] "
           "[--seed N] [--repetition-penalty F] [--greedy] "
           "[--talker-greedy] [--subtalker-greedy] "
           "[--device NAME] [--threads N] "
           "[--output FILE.wav] [--codes-output FILE.bin]\n"
           "       qwen3tts-cli --list-devices\n";
}

const char* device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:
        return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU:
        return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
        return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        return "ACCEL";
    case GGML_BACKEND_DEVICE_TYPE_META:
        return "META";
    }
    return "UNKNOWN";
}

void list_devices() {
    ggml_backend_load_all();

    const size_t count = ggml_backend_dev_count();
    std::cout << "Devices: " << count << '\n';
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);

        constexpr double mib = 1024.0 * 1024.0;
        std::cout << i << ": " << props.name
                  << " | " << device_type_name(props.type)
                  << " | " << props.description;
        if (props.memory_total > 0) {
            std::cout << " | " << std::fixed << std::setprecision(1)
                      << static_cast<double>(props.memory_free) / mib
                      << " / "
                      << static_cast<double>(props.memory_total) / mib
                      << " MiB";
        }
        if (props.device_id != nullptr && props.device_id[0] != '\0') {
            std::cout << " | " << props.device_id;
        }
        std::cout << '\n';
    }
}

#if defined(_WIN32)
std::string utf8_from_wide(const wchar_t* value) {
    if (value == nullptr || value[0] == L'\0') {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        0,
        value,
        -1,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0) {
        return {};
    }
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        value,
        -1,
        result.data(),
        size,
        nullptr,
        nullptr);
    result.pop_back();
    return result;
}
#endif

} // namespace

int run(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    const auto elapsed_text = [](clock::time_point start) {
        const double seconds = std::chrono::duration<double>(
            clock::now() - start).count();
        char text[32];
        std::snprintf(text, sizeof(text), "%.2fs", seconds);
        return std::string(text);
    };
    const auto audio_duration_text = [](const qwen3_tts::AudioBuffer& audio) {
        const int64_t sample_count =
            static_cast<int64_t>(audio.samples.size());
        const int64_t samples_per_second =
            static_cast<int64_t>(audio.sample_rate) * audio.channels;
        if (sample_count <= 0 || samples_per_second <= 0) {
            return std::string("0.000s");
        }
        const double seconds =
            static_cast<double>(sample_count) /
            static_cast<double>(samples_per_second);
        char text[32];
        std::snprintf(text, sizeof(text), "%.3fs", seconds);
        return std::string(text);
    };
    Arguments args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 2;
    }
    if (args.list_devices) {
        list_devices();
        return 0;
    }

    qwen3_tts::Qwen3TTSContextParams context_params;
    context_params.device = args.device;
    context_params.n_threads = args.threads;

    std::cerr << "Loading models...\n";
    const auto load_start = clock::now();
    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Status status = context.load(
        args.talker_model,
        args.speech_model,
        context_params);
    if (status) {
        qwen3_tts::Qwen3TTSModel model;
        status = model.load(context);
        if (!status) {
            std::cerr << status.message() << '\n';
            return 1;
        }
        std::cerr << "Models loaded in " << elapsed_text(load_start) << '\n';

        qwen3_tts::CustomVoiceOptions voice;
        voice.speaker = args.speaker;
        voice.language = args.language;
        voice.instruct = args.instruct;
        qwen3_tts::CodeGenerationOptions options =
            model.generation_defaults();
        if (args.max_new_tokens) {
            options.max_new_tokens = *args.max_new_tokens;
        }
        options.seed = args.seed;
        options.talker->do_sample = !(args.greedy || args.talker_greedy);
        options.code_predictor->do_sample =
            !(args.greedy || args.subtalker_greedy);
        if (args.repetition_penalty) {
            options.talker->repetition_penalty = *args.repetition_penalty;
            options.code_predictor->repetition_penalty =
                *args.repetition_penalty;
        }

        const auto generation_start = clock::now();
        options.log = [&](const std::string& line) {
            std::cerr << '[' << elapsed_text(generation_start)
                      << "] " << line << '\n';
        };

        qwen3_tts::AudioBuffer audio;
        qwen3_tts::SpeechCodeMatrix codes;
        std::cerr << "Generating speech codes...\n";
        status = model.generate_custom_voice_codes(
            context,
            args.text,
            voice,
            options,
            codes);
        if (status) {
            std::cerr << "Generated " << codes.frames
                      << " frames in " << elapsed_text(generation_start)
                      << '\n';
        }
        if (status && !args.codes_output.empty()) {
            const auto codes_start = clock::now();
            std::cerr << "Writing codes...\n";
            std::ofstream file(
                args.codes_output,
                std::ios::binary | std::ios::trunc);
            if (!file) {
                status = qwen3_tts::Status::error(
                    "failed to open codes output: " + args.codes_output);
            } else {
                file.write(
                    reinterpret_cast<const char*>(codes.data.data()),
                    static_cast<std::streamsize>(
                        codes.data.size() * sizeof(int32_t)));
                if (!file) {
                    status = qwen3_tts::Status::error(
                        "failed to write codes output: " +
                        args.codes_output);
                }
            }
            if (status) {
                std::cerr << "Codes written in "
                          << elapsed_text(codes_start) << '\n';
            }
        }
        if (status) {
            const auto decode_start = clock::now();
            std::cerr << "Decoding speech...\n";
            status = model.decode_codes(context, codes, audio);
            if (status) {
                std::cerr << "Speech decoded in "
                          << elapsed_text(decode_start)
                          << " (audio "
                          << audio_duration_text(audio) << ")\n";
            }
        }
        if (status) {
            const auto write_start = clock::now();
            std::cerr << "Writing WAV...\n";
            status = qwen3_tts::write_wav(args.output, audio);
            if (status) {
                std::cerr << "WAV written in "
                          << elapsed_text(write_start) << '\n';
            }
        }
        if (status) {
            std::cout << "Wrote " << args.output << " ("
                      << audio.samples.size() << " samples, "
                      << audio.sample_rate << " Hz, "
                      << audio_duration_text(audio) << ")\n";
        }
    }

    if (!status) {
        std::cerr << status.message() << '\n';
    }
    return status ? 0 : 1;
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    (void) argc;
    (void) argv;
    int wide_argc = 0;
    wchar_t** wide_argv = CommandLineToArgvW(
        GetCommandLineW(),
        &wide_argc);
    if (wide_argv == nullptr || wide_argc <= 0) {
        std::cerr << "failed to read Windows command line\n";
        return 2;
    }

    std::vector<std::string> storage(static_cast<size_t>(wide_argc));
    std::vector<char*> utf8_argv(static_cast<size_t>(wide_argc), nullptr);
    for (int i = 0; i < wide_argc; ++i) {
        storage[static_cast<size_t>(i)] = utf8_from_wide(wide_argv[i]);
        utf8_argv[static_cast<size_t>(i)] =
            storage[static_cast<size_t>(i)].data();
    }
    LocalFree(wide_argv);
    return run(wide_argc, utf8_argv.data());
#else
    return run(argc, argv);
#endif
}
