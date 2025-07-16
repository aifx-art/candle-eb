# HiDream Memory Optimization Suggestions

## Current Memory Issues
The HiDream model is running out of GPU memory because all components are loaded simultaneously, including:
- 48 MoE blocks with 4 experts each (~7.5B parameters)
- 48 attention blocks 
- 49 caption projection layers
- All embedders and final layers

## Immediate Solutions

### 1. **Implement Model CPU Offloading**
```rust
// Add CPU offloading capability to HDModel
impl HDModel {
    pub fn enable_cpu_offload(&mut self) {
        // Move blocks to CPU, only load to GPU when needed
        for block in &mut self.double_stream_blocks {
            block.move_to_cpu();
        }
        for block in &mut self.single_stream_blocks {
            block.move_to_cpu();
        }
    }
    
    pub fn forward_with_offload(&self, ...) -> Result<Tensor> {
        // Load blocks to GPU one at a time during forward pass
        for (i, block) in self.double_stream_blocks.iter().enumerate() {
            block.move_to_gpu();
            let result = block.forward_dual(...);
            block.move_to_cpu(); // Free GPU memory immediately
        }
    }
}
```

### 2. **Reduce MoE Expert Count**
```rust
// In Config::new(), reduce from 4 to 2 experts
num_routed_experts: 2,  // Instead of 4
num_activated_experts: 1, // Instead of 2
```
This would cut MoE memory usage in half.

### 3. **Use Lower Precision**
```rust
// Force FP16 instead of BF16/FP32
let dtype = DType::F16; // Instead of device.bf16_default_to_f32()
```

### 4. **Implement Gradient Checkpointing**
```rust
impl HDModel {
    pub fn enable_gradient_checkpointing(&mut self) {
        self.gradient_checkpointing = true;
    }
}
```

### 5. **Sequential Block Loading**
```rust
impl HDModel {
    pub fn forward_sequential(&self, ...) -> Result<Tensor> {
        let mut x = input;
        
        // Load and unload blocks sequentially
        for i in 0..self.double_stream_blocks.len() {
            // Load block i to GPU
            let block = self.load_block_to_gpu(i)?;
            x = block.forward(...)?;
            // Immediately free GPU memory
            self.unload_block_from_gpu(i)?;
        }
        
        x
    }
}
```

### 6. **Memory-Mapped Model Loading**
The current implementation uses:
```rust
let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
```

Consider lazy loading:
```rust
// Only load weights when actually needed
let vb = VarBuilder::from_lazy_safetensors(&[model_file], dtype, &device)?;
```

## Comparison with Python Reference

The Python implementation uses several memory optimizations:
1. **CPU Offloading**: `model_cpu_offload_seq`
2. **VAE Tiling**: `enable_vae_tiling()`
3. **VAE Slicing**: `enable_vae_slicing()`
4. **Memory Cleanup**: `maybe_free_model_hooks()`

## Recommended Implementation Order

1. **Immediate**: Reduce MoE experts from 4→2
2. **Short-term**: Implement CPU offloading for blocks
3. **Medium-term**: Add gradient checkpointing
4. **Long-term**: Implement full memory management system

## Memory Usage Estimation

Current (all on GPU):
- MoE blocks: ~7.5GB (FP16)
- Attention blocks: ~2GB
- Other components: ~0.5GB
- **Total: ~10GB**

With 2 experts + CPU offloading:
- Active on GPU: ~1GB (one block at a time)
- **Total GPU usage: ~2-3GB**
