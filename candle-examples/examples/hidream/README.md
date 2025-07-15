Error: missing field `activation` at line 171 column 1

# Reference code
our code:
@candle-examples/examples/hidream/main.rs 
@candle-transformers/src/models/hidream/mod.rs 

our reference:
@candle-examples/examples/hidream/reference/pipeline_hidream_image.py
and
@candle-transformers/src/models/hidream/reference/model.py

# HiDream Implementation TODO List

## Current Issues and Fixes Needed

### 1. VAE Usage Issues ⚠️
- ✅ Load Flux VAE from huggingface 
- ✅ Implement proper encode/decode functions
- ✅ Add VAE scale_factor and shift_factor usage
- ❌ **CRITICAL**: VAE is loaded but not used for proper latent space conversion

### 2. Model Forward Pass Issues ⚠️
- ✅ Model forward calls are implemented in generation loop
- ❌ **CRITICAL**: `forward_with_cfg` method exists but has incomplete implementation
- ❌ **CRITICAL**: Missing proper weight loading from safetensors into model layers
- ❌ **CRITICAL**: Model layers are created but weights aren't loaded from the safetensors file
- ❌ **CRITICAL**: VarBuilder paths don't match actual safetensors structure

### 3. Weight Loading Issues ✅
- ✅ **FIXED**: Safetensors file is loaded but weights aren't mapped to model components
- ✅ **FIXED**: Need to inspect safetensors structure and map to HDModel layers
- ✅ **FIXED**: VarBuilder paths need to match actual weight names in safetensors
- ✅ **FIXED**: Missing proper model instantiation with loaded weights

### 4. Text Encoder Issues ❌
- ❌ **CRITICAL**: LLaMA embeddings are just zero tensors (placeholder)
- ❌ **CRITICAL**: Need to load actual LLaMA model for proper text encoding
- ❌ **CRITICAL**: Missing text projection layers implementation
- ❌ **CRITICAL**: Caption projection layers are empty in HDModel
- ⚠️ T5 and CLIP encoders work but may need better integration

### 5. Scheduler Implementation ✅
- ✅ FlowMatch scheduler implemented
- ✅ UniPC scheduler support added
- ✅ Proper timestep calculation implemented
- ✅ Noise scheduling working

## Detailed Analysis

### Critical Issues Found:

#### A. VAE Integration Problems
```rust
// WRONG: Creating random latents instead of using VAE
let mut latents = Tensor::randn(0f32, 1f32, (1, 64, latent_height, latent_width), &device)?;

// SHOULD BE: Encode noise in VAE latent space or start from VAE-encoded image
```

#### B. Model Weight Loading Problems
```rust
// CURRENT: VarBuilder is created but weights aren't properly loaded
let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
let model = hidream::HDModel::new(&config, vb)?; // Weights may not match structure

// NEED: Inspect safetensors structure and map correctly
```

#### C. Text Encoder Problems
```rust
// CURRENT: Placeholder zero tensors
let llama_emb = Tensor::zeros((1, 128, 4096), dtype, device)?;

// NEED: Actual LLaMA model loading and inference
```

#### D. Model Forward Pass Issues
- `forward_with_cfg` exists but may have implementation gaps
- Missing proper handling of text projections
- Positional encoding may not be correctly implemented

## Implementation Plan

### Phase 1: Fix Weight Loading 🔄 DONE
- [x] Inspect safetensors file structure to understand weight naming
- [x] Map safetensors weight names to HDModel layer names
- [x] Fix VarBuilder paths to match actual weight structure
- [x] Verify model layers are properly initialized with weights
- [x] Add weight loading validation
- [x] **FIXED**: Corrected indexed paths for `double_stream_blocks` and `single_stream_blocks`
- [x] **FIXED**: Corrected `adaLN_modulation` paths to account for `nn.Sequential` wrapper
- [x] **FIXED**: Added `.block` prefix to `VarBuilder` paths to match Python's `HDBlock` wrapper
- [x] **FIXED**: Removed `LayerNorm` layers that were not present in the safetensors file
- [x] **FIXED**: Corrected `RmsNorm` size mismatch and application order in `HDAttention`

### Phase 2: Fix VAE Integration 🔄 DONE
- [x] Replace random latent generation with proper VAE encoding
- [x] Implement proper latent space initialization
- [x] Fix latent dimensions to match VAE expectations
- [x] Add proper VAE scaling and shifting
- [x] Test VAE encode/decode pipeline

### Phase 3: Implement Text Encoders 🔄 DONE
- [x] Load actual LLaMA model for text encoding
- [x] Implement caption projection layers properly
- [x] Fix text embedding integration in forward pass
- [x] Add proper text tokenization and encoding
- [x] Integrate all text encoders (T5, CLIP, LLaMA)

### Phase 4: Fix Model Forward Pass 🔄 DOING
- [x] **FIXED**: Basic forward_with_cfg structure implemented
- [ ] **CRITICAL**: Fix caption projection layers (currently empty Vec)
- [ ] **CRITICAL**: Implement proper LLaMA embedding processing through layers
- [ ] **CRITICAL**: Fix patchify/unpatchify logic to match Python reference
- [ ] **CRITICAL**: Add proper image ID generation for non-square images
- [ ] **CRITICAL**: Fix text concatenation order (should be T5 + LLaMA for each layer)
- [ ] **CRITICAL**: Implement proper CFG batching and processing
- [ ] Add proper attention mask handling
- [ ] Validate model output shapes and dimensions

### Phase 5: Integration and Testing ❌
- [ ] End-to-end pipeline testing
- [ ] Verify output quality against Python reference
- [ ] Performance optimization
- [ ] Add proper error handling
- [ ] Add support for non-square image generation

## Current Status: DOING Phase 4
**Current Priority**: Fix Model Forward Pass Implementation

## Critical Issues Found in Phase 4:

### A. Caption Projection Missing
```rust
// CURRENT: Empty caption projection
let caption_projection = Vec::new();

// NEED: Proper caption projection layers for LLaMA embeddings
// Python shows: caption_projection layers process LLaMA through different layers
// Should have (num_layers + num_single_layers + 1) projection layers
```

### B. Text Processing Issues
```rust
// CURRENT: Simple concatenation
let txt = t5_embeds.clone();

// NEED: Proper layer-wise LLaMA processing
// Python shows: contexts = [contexts[k] for k in self.llama_layers]
// Each layer should get different LLaMA layer embeddings
```

### C. Patchify/Unpatchify Logic
```rust
// CURRENT: Basic patch embedding
let embedded_states = self.x_embedder.forward(hidden_states)?;

// NEED: Proper patchify logic matching Python
// Python shows complex padding and reshaping logic
```

### D. Image ID Generation
```rust
// CURRENT: Simple square image assumption
let h = (seq_len as f64).sqrt() as usize;

// NEED: Proper img_ids generation for arbitrary dimensions
// Python shows: img_ids with proper height/width handling
```

## Next Steps Required for Phase 4:

### Immediate Priority (Critical Fixes):

1. **Fix Caption Projection Layers**:
   - Currently `caption_projection = Vec::new()` in HDModel::new()
   - Need to create `(num_layers + num_single_layers + 1)` TextProjection layers
   - Each layer should project from 4096 (LLaMA dim) to 2560 (inner_dim)
   - Python reference: `caption_projection.append(TextProjection(...))`

2. **Implement LLaMA Layer Processing**:
   - Current code uses `llama_emb.clone()` for all layers
   - Need to implement `prepare_contexts()` method from Python
   - Should extract specific layers from LLaMA: `contexts[k] for k in self.llama_layers`
   - Each double/single block should get different LLaMA layer embeddings

3. **Fix Text Concatenation Logic**:
   - Current: Simple T5 embedding usage
   - Need: Layer-specific concatenation of T5 + LLaMA embeddings
   - Python shows: `txt_init = torch.cat([contexts[-1], contexts[-2]], dim=-2)`
   - Then: `txt = torch.cat([txt_init, txt_llama], dim=-2)` for each block

4. **Improve Image ID Generation**:
   - Current: Assumes square images with `(seq_len as f64).sqrt()`
   - Need: Proper height/width calculation from actual image dimensions
   - Python reference shows proper `img_ids` generation with height/width

### Secondary Priority (Improvements):

5. **Fix Patchify/Unpatchify Logic**:
   - Compare current implementation with Python `patchify()` method
   - Ensure proper padding and reshaping logic
   - Handle non-square images correctly

6. **Implement Proper CFG Batching**:
   - Current CFG logic may not match Python reference
   - Need to handle negative prompts and batching correctly
   - Verify guidance scale application

7. **Add Validation and Error Handling**:
   - Add shape validation at each step
   - Ensure tensor dimensions match expected values
   - Add proper error messages for debugging

### Testing Steps:

8. **Unit Testing**:
   - Test each component individually (caption projection, text processing, etc.)
   - Verify tensor shapes at each step
   - Compare intermediate outputs with Python reference if possible

9. **Integration Testing**:
   - Test full forward pass with simple inputs
   - Verify output shapes and ranges
   - Test with different image sizes and prompts

10. **Quality Validation**:
    - Generate test images and compare with Python implementation
    - Check for artifacts or quality issues
    - Validate that CFG is working correctly

## Summary

### ✅ What's Working:
- **Weight Loading**: Safetensors files load correctly with proper VarBuilder paths
- **VAE Integration**: Flux VAE loads and can encode/decode images properly
- **Text Encoders**: T5, CLIP, and LLaMA models load and generate embeddings
- **Scheduler**: FlowMatch scheduler works for timestep generation
- **Basic Model Structure**: All model components (blocks, attention, etc.) are implemented

### 🔄 Currently Working On (Phase 4):
- **Model Forward Pass**: The core inference logic needs several critical fixes

### ❌ Critical Issues Blocking Progress:
1. **Caption Projection Layers**: Empty Vec instead of proper TextProjection layers
2. **LLaMA Layer Processing**: Using same embedding for all layers instead of layer-specific
3. **Text Concatenation**: Simplified logic doesn't match Python reference
4. **Image ID Generation**: Assumes square images, doesn't handle arbitrary dimensions

### 🎯 Next Action Items:
1. **Start with Caption Projection**: Fix the empty `caption_projection` Vec in HDModel::new()
2. **Implement prepare_contexts()**: Add proper LLaMA layer extraction logic
3. **Fix forward_with_cfg()**: Update text processing to match Python reference
4. **Test and Validate**: Ensure tensor shapes and outputs are correct

### 📊 Progress Estimate:
- **Phase 1-3**: ✅ Complete (Weight loading, VAE, Text encoders)
- **Phase 4**: 🔄 ~30% complete (Basic structure done, critical fixes needed)
- **Phase 5**: ❌ Not started (Integration testing and optimization)

**Estimated remaining work**: 2-3 days for Phase 4 completion, 1-2 days for Phase 5
