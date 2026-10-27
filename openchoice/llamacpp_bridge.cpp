// Pinned llama.cpp C ABI: exact final-token logits, no sampling or generation.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int context_tokens = 4096;
constexpr int microbatch_tokens = 512;
constexpr int cancelled = -200;
struct Session {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    mtmd_context * vision = nullptr;
    mtmd_input_chunks * chunks = nullptr;
    std::string projector;
    int threads = 12;
    ggml_abort_callback abort = nullptr;
    void * abort_data = nullptr;
    std::vector<uint8_t> prefix;
    int prefix_tokens = 0;
    int expanded_tokens = 0;
    std::vector<std::pair<std::string, std::vector<float>>> images;
    ~Session() {
        if (chunks) mtmd_input_chunks_free(chunks);
        if (vision) mtmd_free(vision);
        if (batch.token) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    bool stopped() const { return abort && abort(abort_data); }
};
void private_log(ggml_log_level, const char *, void *) {
    // Native diagnostics can contain request data, including on normal aborts.
    // Failures propagate through C ABI status codes; only GPU state transitions
    // are logged by Python. Do not emit per-request backend diagnostics.
}
void initialize() {
    static std::once_flag init;
    std::call_once(init, [] {
        llama_log_set(private_log, nullptr);
        ggml_log_set(private_log, nullptr);
        mtmd_helper_log_set(private_log, nullptr);
        llama_backend_init();
    });
}
int read_logits(Session & s, const int32_t * answers, int count, float * output) {
    if (count < 0 || count > 26) return -100;
    const auto * logits = llama_get_logits_ith(s.ctx, -1);
    if (!logits) return -101;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(s.model));
    for (int i = 0; i < count; ++i) {
        if (answers[i] < 0 || answers[i] >= vocab) return -100;
        output[i] = logits[answers[i]];
    }
    return 0;
}
void clear(Session & s) {
    if (s.chunks) { mtmd_input_chunks_free(s.chunks); s.chunks = nullptr; }
    s.expanded_tokens = 0;
    std::fill(s.prefix.begin(), s.prefix.end(), 0);
    s.prefix.clear();
    s.prefix_tokens = 0;
    for (auto & entry : s.images) std::fill(entry.second.begin(), entry.second.end(), 0);
    s.images.clear();
    std::fill(s.batch.token, s.batch.token + context_tokens, 0);
    llama_memory_clear(llama_get_memory(s.ctx), true);
}
}

extern "C" {
int oc_abi_version() { return 1; }
int oc_has_gpu() { initialize(); return llama_supports_gpu_offload(); }
void * oc_open(const char * model, const char * projector, int threads, int gpu_layers,
               ggml_abort_callback abort, void * abort_data) {
    try {
        initialize();
        if (threads < 1 || gpu_layers < 0 || (gpu_layers && !llama_supports_gpu_offload())) return nullptr;
        auto s = std::make_unique<Session>();
        s->threads = threads;
        s->projector = projector;
        s->abort = abort;
        s->abort_data = abort_data;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        mp.use_extra_bufts = false;
        s->model = llama_model_load_from_file(model, mp);
        if (!s->model) return nullptr;
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = context_tokens;
        cp.n_ubatch = microbatch_tokens;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.offload_kqv = cp.op_offload = gpu_layers > 0;
        cp.no_perf = true;
        cp.abort_callback = abort;
        cp.abort_callback_data = abort_data;
        s->ctx = llama_init_from_model(s->model, cp);
        if (!s->ctx) return nullptr;
        s->batch = llama_batch_init(context_tokens, 0, 1);
        return s.release();
    } catch (...) { return nullptr; }
}
void oc_close(void * handle) { delete static_cast<Session *>(handle); }
void oc_clear(void * handle) { clear(*static_cast<Session *>(handle)); }
int oc_eval(void * handle, const int32_t * tokens, int count, int position, int reset,
            const int32_t * answers, int answer_count, float * output) {
    auto & s = *static_cast<Session *>(handle);
    if (count <= 0 || position < 0 || count + position > context_tokens) return -100;
    if (reset) llama_memory_clear(llama_get_memory(s.ctx), true);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(s.model));
    // The same 512-token boundaries as a full llama_decode; also bounds CUDA
    // cancellation latency, because upstream's abort callback is CPU-only.
    for (int start = 0; start < count; start += microbatch_tokens) {
        if (s.stopped()) return cancelled;
        const int n = std::min(microbatch_tokens, count - start);
        s.batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            const int token = tokens[start + i];
            if (token < 0 || token >= vocab) return -100;
            s.batch.token[i] = token;
            s.batch.pos[i] = position + start + i;
            s.batch.n_seq_id[i] = 1;
            s.batch.seq_id[i][0] = 0;
            s.batch.logits[i] = answer_count > 0 && start + i + 1 == count;
        }
        const int status = llama_decode(s.ctx, s.batch);
        if (s.stopped()) return cancelled;
        if (status) return status;
    }
    return answer_count > 0 ? read_logits(s, answers, answer_count, output) : 0;
}
size_t oc_save_prefix(void * handle, int count) {
    auto & s = *static_cast<Session *>(handle);
    if (count <= 0 || count % microbatch_tokens || count >= context_tokens) return 0;
    try {
        s.prefix.resize(llama_state_seq_get_size(s.ctx, 0));
        const auto written = llama_state_seq_get_data(s.ctx, s.prefix.data(), s.prefix.size(), 0);
        if (written != s.prefix.size()) return 0;
        s.prefix_tokens = count;
        return written;
    } catch (...) { return 0; }
}
int oc_restore_prefix(void * handle) {
    auto & s = *static_cast<Session *>(handle);
    if (s.prefix.empty()) return -1;
    llama_memory_clear(llama_get_memory(s.ctx), true);
    return llama_state_seq_set_data(s.ctx, s.prefix.data(), s.prefix.size(), 0) == s.prefix.size()
        ? s.prefix_tokens : -1;
}
int oc_media_prepare(void * handle, const char * text, const char * const * paths,
                     const char * const * hashes, int image_count, int image_token,
                     int32_t * expanded, int capacity) {
    auto & s = *static_cast<Session *>(handle);
    try {
        if (s.stopped()) return cancelled;
        if (!s.vision) {
            auto params = mtmd_context_params_default();
            // The projector always stays on CPU; GPU-auto is a text-serving policy.
            params.use_gpu = false;
            params.n_threads = s.threads;
            params.warmup = params.print_timings = false;
            // Exact HF Qwen3.5 image processor limits (32x32 pixels per token).
            params.image_min_tokens = 64;
            params.image_max_tokens = 16384;
            s.vision = mtmd_init_from_file(s.projector.c_str(), s.model, params);
            if (!s.vision || !mtmd_decode_use_mrope(s.vision)) return -102;
        }
        if (s.chunks) mtmd_input_chunks_free(s.chunks);
        s.chunks = mtmd_input_chunks_init();
        s.expanded_tokens = 0;
        if (s.images.empty()) s.images.reserve(image_count);
        using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
        std::vector<Bitmap> owned;
        std::vector<const mtmd_bitmap *> bitmaps;
        owned.reserve(image_count);
        bitmaps.reserve(image_count);
        for (int i = 0; i < image_count; ++i) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(s.vision, paths[i], false, mtmd_helper_init_opt_default());
            owned.emplace_back(wrapper.bitmap, mtmd_bitmap_free);
            if (!wrapper.bitmap || wrapper.video_ctx) return -103;
            mtmd_bitmap_set_id(wrapper.bitmap, hashes[i]);
            bitmaps.push_back(wrapper.bitmap);
        }
        mtmd_input_text input{text, std::strlen(text), false, true};
        const int status = mtmd_tokenize(s.vision, s.chunks, &input, bitmaps.data(), bitmaps.size());
        if (status) return -104;
        for (size_t i = 0; i < mtmd_input_chunks_size(s.chunks); ++i) {
            const auto * chunk = mtmd_input_chunks_get(s.chunks, i);
            const auto kind = mtmd_input_chunk_get_type(chunk);
            size_t n = 0;
            if (kind == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
                if (n > static_cast<size_t>(capacity - s.expanded_tokens)) return -106;
                std::copy_n(tokens, n, expanded + s.expanded_tokens);
            } else if (kind == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                n = mtmd_input_chunk_get_n_tokens(chunk);
                if (n > static_cast<size_t>(capacity - s.expanded_tokens)) return -106;
                std::fill_n(expanded + s.expanded_tokens, n, image_token);
            } else return -105;
            s.expanded_tokens += static_cast<int>(n);
        }
        return s.expanded_tokens;
    } catch (const std::bad_alloc &) { return -3; }
      catch (...) { return -107; }
}
int oc_media_eval(void * handle, const int32_t * answers, int count, float * output) {
    auto & s = *static_cast<Session *>(handle);
    try {
        if (!s.chunks || s.expanded_tokens > context_tokens) return -100;
        llama_memory_clear(llama_get_memory(s.ctx), true);
        llama_pos past = 0;
        const size_t chunks = mtmd_input_chunks_size(s.chunks);
        for (size_t i = 0; i < chunks; ++i) {
            if (s.stopped()) return cancelled;
            const auto * chunk = mtmd_input_chunks_get(s.chunks, i);
            int status;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const char * id = mtmd_input_chunk_get_id(chunk);
                if (!id) return -108;
                auto found = std::find_if(s.images.begin(), s.images.end(),
                    [id](const auto & image) { return image.first == id; });
                if (found == s.images.end()) {
                    status = mtmd_encode_chunk(s.vision, chunk);
                    if (status) return status;
                    if (s.stopped()) return cancelled;
                    const size_t size = llama_model_n_embd_inp(s.model) * mtmd_input_chunk_get_n_tokens(chunk);
                    const float * data = mtmd_get_output_embd(s.vision);
                    s.images.emplace_back(id, std::vector<float>(data, data + size));
                    found = s.images.end() - 1;
                }
                status = mtmd_helper_decode_image_chunk(s.vision, s.ctx, chunk,
                    found->second.data(), past, 0, microbatch_tokens, &past, nullptr, nullptr);
            } else {
                status = mtmd_helper_eval_chunk_single(s.vision, s.ctx, chunk,
                    past, 0, microbatch_tokens, i + 1 == chunks, &past);
            }
            if (s.stopped()) return cancelled;
            if (status) return status;
        }
        return read_logits(s, answers, count, output);
    } catch (const std::bad_alloc &) { return -3; }
      catch (...) { return -109; }
}
}
