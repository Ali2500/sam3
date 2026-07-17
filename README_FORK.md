# FAR Fork — Changes vs. Upstream SAM 3.1

This is a fork of [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3),
branched from the **SAM 3.1 Release** (`20dba30`). It carries a small set of
changes needed to run SAM 3.1 video tracking at scale inside FAR's data-annotation
pipeline — primarily **long-video inference** (which the stock model cannot do
because GPU memory grows linearly with video length), plus a few environment and
correctness fixes.

All changes live on the `sam3.1` branch on top of the upstream release commit, so
the delta is exactly one commit:

```bash
git diff 20dba30 sam3.1        # full diff of every change described below
```

## Why this fork exists

The upstream video predictor keeps every frame's tracking state
(`maskmem_features`, `pred_masks`, backbone features) resident on the GPU for the
whole clip. On the multi-minute web videos we annotate, this OOMs long before the
clip finishes. The bulk of the changes below implement **opt-in CPU offloading**
of that state, plus the cache eviction and device-management needed to make it
correct. The remainder adapt the model to our runtime: S3 I/O, an A100 (non-Hopper)
attention path, and Ray-aware logging.

---

## 1. CPU offloading for long-video inference

**Cause:** GPU memory grew linearly with video length, OOMing on long clips. These
changes let per-frame tracking state be offloaded to CPU (opt-in via
`offload_state_to_cpu`), trading ~10% throughput for bounded GPU memory.

| File | Change |
|------|--------|
| `sam3/model/data_misc.py` | Added `NestedTensor.cpu()` convenience method so nested tensors can be moved to CPU alongside plain tensors. |
| `sam3/model_builder.py` | Added `offload_state_to_cpu: bool = False` parameter (with docstring) to `build_sam3_multiplex_video_predictor`, threaded into the predictor wrapper. |
| `sam3/model/sam3_base_predictor.py` | Plumbed `offload_state_to_cpu` through the request handler and `start_session` into the inference state. |
| `sam3/model/sam3_multiplex_base.py` | `Sam3MultiplexPredictorWrapper` stores `offload_state_to_cpu`; passed to all three `init_state` calls that build per-object tracker states. |
| `sam3/model/sam3_video_base.py` | Forward the tracker's `offload_state_to_cpu` flag when initializing a new tracker state. |
| `sam3/model/sam3_multiplex_tracking.py` | `init_state` accepts and records `offload_state_to_cpu`; `_init_new_sam2_state` forwards it. Cached per-frame masks are moved to CPU in `_cache_frame_outputs`; `_build_sam2_output` moves refined masks back to the cache's device (and returns refined masks directly when no cache entry exists). Accumulated `grounding_cache` / `multigpu_buffer` and the detector backbone cache are evicted each frame to prevent GPU OOM. |
| `sam3/model/sam3_image.py` | Added an optional GPU/CPU backbone-feature cache (`enable_/disable_/clear_/move_cache_entries_to_cpu/evict_from_backbone_cache`) plus a `struct_to` helper. `Sam3Image.forward` reuses cached backbone features across text prompts for the same frames instead of recomputing them. |
| `sam3/model/video_tracking_multiplex_demo.py` | When reusing an existing frame output, move any offloaded (CPU) tensors back to the inference device before in-place modification, avoiding a device mismatch. |

## 2. Reverse-propagation / bidirectional-tracking bug fixes

**Cause:** Bugs surfaced while running bidirectional propagation on our clips
(off-by-one on the reverse pass, and stale tracking buffers leaking between the
forward and backward passes).

| File | Change |
|------|--------|
| `sam3/model/sam3_multiplex_tracking.py` | Fixed an off-by-one in the reverse `processing_order`: `range(start_frame_idx - 1, ...)` → `range(start_frame_idx, ...)` so the start frame is not skipped on the backward pass. |
| `sam3/model/sam3_base_predictor.py` | In `propagate_in_video`: skip the forward pass when already at the last frame and the backward pass when already at the first frame; and reset temporal tracking buffers (`sam2_inference_states`, `feature_cache`, `cached_frame_outputs`, `tracker_metadata`) between directions for a `"both"` run so the backward pass starts fresh. Guarded by a new `reset_tracker_metadata` flag (default `True`; point prompts pass `False` since they handle bidirectional propagation at the interface level). |

## 3. S3 / remote I/O support

**Cause:** In our pipeline, videos, checkpoints, and the BPE vocab live on S3, not
the local filesystem. `smart_open` handles `s3://` URIs transparently.

| File | Change |
|------|--------|
| `sam3/model/io_utils.py` | The cv2 video loader reads bytes via `smart_open` into a temp file before `cv2.VideoCapture`, so S3-hosted videos load without a local copy. (torchcodec path explicitly asserts non-S3.) |
| `sam3/model/tokenizer_ve.py` | BPE vocab is opened with `smart_open` for `s3://` paths, with a `gzip.BadGzipFile` fallback for already-decompressed files. |
| `sam3/model_builder.py` | Checkpoint loading (`_load_checkpoint`, both video builders, and the multiplex predictor) opens `s3://` paths via `smart_open`. |

## 4. `max_frames_to_load` cap

**Cause:** Needed to bound decode cost / cap clip length for long or multi-hour
source videos.

| File | Change |
|------|--------|
| `sam3/model/io_utils.py` | Added a `max_frames_to_load` parameter threaded through `load_resource_as_video_frames`, `load_video_frames`, and the image-folder and cv2 loaders (torchcodec asserts it is unset). |

## 5. Environment adaptations

**Cause:** Our runtime differs from the upstream reference environment (A100 rather
than Hopper GPUs, Ray-based distribution, and a newer packaging API).

| File | Change | Reason |
|------|--------|--------|
| `sam3/perflib/fa3.py` | Attention dtype `float8_e4m3fn` → `bfloat16` (float8 kept commented), and import path `flash_attn_interface` → `flash_attn.flash_attn_interface`. | `float8_e4m3fn` FA3 requires Hopper (H100); we run on A100 and older GPUs, which needs bfloat16 and the `flash_attn`-packaged interface. |
| `sam3/model/utils/misc.py` | Added `is_ray_initialized()` with a guarded `import ray` (returns `False` if Ray is not installed). | Suppress duplicate `tqdm` progress bars across Ray workers. Ray stays an optional dependency, so this remains upstream-friendly. |
| `sam3/model/sam3_video_inference.py`, `sam3/model/sam3_multiplex_tracking.py` | Progress-bar `disable=self.rank > 0` → `disable=self.rank > 0 or is_ray_initialized()`. | The DDP-rank check does not dedupe across Ray worker *processes*; each is its own process. Combining both covers DDP and Ray. |
| `sam3/model_builder.py` | `pkg_resources.resource_filename(...)` → `importlib.resources.files(...)` for the BPE asset path. | `pkg_resources` is deprecated; `importlib.resources` is the supported API. (Generally upstreamable.) |

---

## Notes for upstreaming

Several changes are cleanly separable and could be proposed upstream on their own:

- the `pkg_resources` → `importlib.resources` migration (§5),
- the reverse-propagation bug fixes (§2),
- the optional `offload_state_to_cpu` flag (§1).

The S3/`smart_open` I/O (§3) and the A100 attention dtype (§5) are more
environment-specific and are more likely to stay fork-local.
