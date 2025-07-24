# Candle EB - Extended Candle with HiDream Support

This is an extended version of the Candle deep learning framework with added support for HiDream models.

## HiDream Implementation Status

### TODO
- [ ] Test quantized model loading and inference
- [ ] Add proper error handling for missing quantized features
- [ ] Verify model configuration matches GGUF format expectations
- [ ] Add support for different HiDream model variants in quantized format

### DOING

### DONE
- [x] Add HiDream model module structure
- [x] Implement basic HiDream model components (attention, feed forward, MoE)
- [x] Add HiDream configuration and model loading
- [x] Create quantized model file with all necessary components
- [x] Add text projection layers for LLaMA and T5 embeddings
- [x] Implement double and single stream blocks for quantized model
- [x] Implement HiDream quantized model structure
- [x] Add quantized attention, feed forward, and MoE components
- [x] Create HDQuantizedModel with proper forward_with_cfg method
- [x] Update main.rs to use quantized model when --quantized flag is set
- [x] Add GGUF model loading support for HiDream quantized models

## Features

- Support for HiDream I1 and E1 models
- Quantized model support using GGUF format
- Multiple model variants (FP8, FP16, BF16)
- Text-to-image generation
- Image editing capabilities (E1 models)

## Usage

```bash
# Generate image with quantized model
cargo run --example hidream --features quantized -- --quantized --prompt "A cat holding a sign"

# Generate image with regular model
cargo run --example hidream -- --prompt "A cat holding a sign"
```

## Model Variants

- HiDream-I1-Fast-FP8: Fastest generation (16 steps, no CFG)
- HiDream-I1-Dev-FP8: Fast generation (28 steps, no CFG)  
- HiDream-I1-Full-FP8: High quality (50 steps, CFG 5.0)
- HiDream-E1-Full-BF16: Image editing model

## GGUF Models

Quantized GGUF models are available at: https://huggingface.co/ND911/hidream_i1_fp8_full_dev_fast_ggufs
