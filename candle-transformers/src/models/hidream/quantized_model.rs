use super::{timestep_embedding, EmbedNd, Config};
use crate::quantized_nn::{linear, linear_b, Linear};
use crate::quantized_var_builder::VarBuilder;
use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::layer_norm::RmsNormNonQuantized;
use candle_nn::{LayerNorm, RmsNorm};

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, DType::F32, vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        let in_layer = linear(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = linear(h_sz, h_sz, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl candle::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm<RmsNormNonQuantized>,
    key_norm: RmsNorm<RmsNormNonQuantized>,
}

impl QkNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?.dequantize(vb.device())?;
        let query_norm = RmsNorm::<RmsNormNonQuantized>::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?.dequantize(vb.device())?;
        let key_norm = RmsNorm::<RmsNormNonQuantized>::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

// HiDream-specific modulation for double stream (12 components)
#[derive(Debug, Clone)]
struct Modulation12 {
    lin: Linear,
}

impl Modulation12 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = linear(dim, 12 * dim, vb.pp("adaLN_modulation.1"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<[Tensor; 12]> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(12, D::Minus1)?;
        if ys.len() != 12 {
            candle::bail!("unexpected len from chunk {ys:?}")
        }
        Ok([
            ys[0].clone(), ys[1].clone(), ys[2].clone(), ys[3].clone(),
            ys[4].clone(), ys[5].clone(), ys[6].clone(), ys[7].clone(),
            ys[8].clone(), ys[9].clone(), ys[10].clone(), ys[11].clone(),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct HDAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    to_q_t: Option<Linear>,
    to_k_t: Option<Linear>,
    to_v_t: Option<Linear>,
    to_out_t: Option<Linear>,
    q_norm: QkNorm,
    k_norm: QkNorm,
    q_norm_t: Option<QkNorm>,
    k_norm_t: Option<QkNorm>,
    num_heads: usize,
    head_dim: usize,
    single: bool,
}

impl HDAttention {
    fn new(
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        single: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = num_heads * head_dim;

        let to_q = linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(inner_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(inner_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, dim, vb.pp("to_out"))?;

        let q_norm = QkNorm::new(head_dim, vb.pp("q_rms_norm"))?;
        let k_norm = QkNorm::new(head_dim, vb.pp("k_rms_norm"))?;

        let (to_q_t, to_k_t, to_v_t, to_out_t, q_norm_t, k_norm_t) = if single {
            (None, None, None, None, None, None)
        } else {
            let to_q_t = linear(dim, inner_dim, vb.pp("to_q_t"))?;
            let to_k_t = linear(inner_dim, inner_dim, vb.pp("to_k_t"))?;
            let to_v_t = linear(inner_dim, inner_dim, vb.pp("to_v_t"))?;
            let to_out_t = linear(inner_dim, dim, vb.pp("to_out_t"))?;
            let q_norm_t = QkNorm::new(head_dim, vb.pp("q_rms_norm_t"))?;
            let k_norm_t = QkNorm::new(head_dim, vb.pp("k_rms_norm_t"))?;
            (
                Some(to_q_t),
                Some(to_k_t),
                Some(to_v_t),
                Some(to_out_t),
                Some(q_norm_t),
                Some(k_norm_t),
            )
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            to_q_t,
            to_k_t,
            to_v_t,
            to_out_t,
            q_norm,
            k_norm,
            q_norm_t,
            k_norm_t,
            num_heads,
            head_dim,
            single,
        })
    }

    fn forward_dual(&self, img: &Tensor, txt: &Tensor, pe: &Tensor) -> Result<(Tensor, Tensor)> {
        // Image stream
        let q_i = self.to_q.forward(img)?;
        let k_i = self.to_k.forward(img)?;
        let v_i = self.to_v.forward(img)?;

        // Text stream
        let q_t = self.to_q_t.as_ref().unwrap().forward(txt)?;
        let k_t = self.to_k_t.as_ref().unwrap().forward(txt)?;
        let v_t = self.to_v_t.as_ref().unwrap().forward(txt)?;

        let (b, seq_i, _) = img.dims3()?;
        let (_, seq_t, _) = txt.dims3()?;
        
        let q_i = q_i.reshape((b, seq_i, self.num_heads, self.head_dim))?;
        let k_i = k_i.reshape((b, seq_i, self.num_heads, self.head_dim))?;
        let v_i = v_i.reshape((b, seq_i, self.num_heads, self.head_dim))?;
        
        let q_t = q_t.reshape((b, seq_t, self.num_heads, self.head_dim))?;
        let k_t = k_t.reshape((b, seq_t, self.num_heads, self.head_dim))?;
        let v_t = v_t.reshape((b, seq_t, self.num_heads, self.head_dim))?;

        // Apply RMS norm
        let q_i = q_i.apply(&self.q_norm.query_norm)?;
        let k_i = k_i.apply(&self.k_norm.key_norm)?;
        let q_t = q_t.apply(&self.q_norm_t.as_ref().unwrap().query_norm)?;
        let k_t = k_t.apply(&self.k_norm_t.as_ref().unwrap().key_norm)?;

        // Concatenate for joint attention
        let q = Tensor::cat(&[q_i, q_t], 1)?;
        let k = Tensor::cat(&[k_i, k_t], 1)?;
        let v = Tensor::cat(&[v_i, v_t], 1)?;

        // Apply attention with positional encoding
        let attn_out = super::attention(&q, &k, &v, pe)?;
        
        // Split back to image and text streams
        let img_out = attn_out.narrow(1, 0, seq_i)?;
        let txt_out = attn_out.narrow(1, seq_i, seq_t)?;

        // Apply output projections
        let img_out = img_out.reshape((b, seq_i, self.num_heads * self.head_dim))?;
        let txt_out = txt_out.reshape((b, seq_t, self.num_heads * self.head_dim))?;

        let img_out = self.to_out.forward(&img_out)?;
        let txt_out = self.to_out_t.as_ref().unwrap().forward(&txt_out)?;

        Ok((img_out, txt_out))
    }

    fn forward_single(&self, x: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        
        let (b, seq, _) = x.dims3()?;
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, seq, self.num_heads, self.head_dim))?;

        let q = q.apply(&self.q_norm.query_norm)?;
        let k = k.apply(&self.k_norm.key_norm)?;

        let attn_out = super::attention(&q, &k, &v, pe)?;
        let attn_out = attn_out.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.to_out.forward(&attn_out)
    }
}

// HiDream Feed Forward with SwiGLU
#[derive(Debug, Clone)]
struct HDFeedForwardSwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl HDFeedForwardSwiGLU {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Apply the same transformation as in Python reference:
        // hidden_dim = int(2 * hidden_dim / 3)
        // hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        let multiple_of = 256;
        let mut actual_hidden_dim = (2 * hidden_dim) / 3;
        actual_hidden_dim = multiple_of * ((actual_hidden_dim + multiple_of - 1) / multiple_of);
        
        let w1 = linear(dim, actual_hidden_dim, vb.pp("w1"))?;
        let w2 = linear(actual_hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear(dim, actual_hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl candle::Module for HDFeedForwardSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.w1.forward(x)?.silu()?;
        let b = self.w3.forward(x)?;
        (a * b)?.apply(&self.w2)
    }
}

// HiDream MoE Gate
#[derive(Debug, Clone)]
struct HDMoEGate {
    weight: Tensor,
    top_k: usize,
}

impl HDMoEGate {
    fn new(dim: usize, num_routed_experts: usize, num_activated_experts: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((num_routed_experts, dim), "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, top_k: num_activated_experts })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = x.matmul(&self.weight.t()?)?;
        let scores = candle_nn::ops::softmax(&logits, D::Minus1)?;
        super::topk(&scores, self.top_k, -1)
    }
}

// HiDream MoE Feed Forward
#[derive(Debug, Clone)]
struct HDMOEFeedForwardSwiGLU {
    shared_experts: HDFeedForwardSwiGLU,
    experts: Vec<HDFeedForwardSwiGLU>,
    gate: HDMoEGate,
    num_activated_experts: usize,
}

impl HDMOEFeedForwardSwiGLU {
    fn new(
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_activated_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let shared_experts = HDFeedForwardSwiGLU::new(dim, hidden_dim / 2, vb.pp("shared_experts"))?;
        let mut experts = Vec::with_capacity(num_routed_experts);
        for i in 0..num_routed_experts {
            experts.push(HDFeedForwardSwiGLU::new(dim, hidden_dim, vb.pp(&format!("experts.{}", i)))?);
        }
        let gate = HDMoEGate::new(dim, num_routed_experts, num_activated_experts, vb.pp("gate"))?;
        Ok(Self { shared_experts, experts, gate, num_activated_experts })
    }
}

impl candle::Module for HDMOEFeedForwardSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y_shared = self.shared_experts.forward(x)?;
        let (topk_weight, topk_idx) = self.gate.forward(x)?;
        let tk_idx_flat = topk_idx.flatten_all()?.to_dtype(DType::U32)?;
        let x_repeated = super::repeat_interleave(x, self.num_activated_experts, 0)?;
        let mut y = Tensor::zeros(x_repeated.shape(), x.dtype(), x.device())?;
        for (i, expert) in self.experts.iter().enumerate() {
            let mask = tk_idx_flat.eq(&Tensor::new(i as u32, x.device())?)?;
            let x_sel = super::mask_where(&x_repeated, &mask.broadcast_as(x_repeated.shape())?, &Tensor::zeros(x_repeated.shape(), x.dtype(), x.device())?)?;
            if x_sel.dim(0)? == 0 {
                continue;
            }
            let expert_out = expert.forward(&x_sel)?;
            y = super::masked_fill(&y, &mask.broadcast_as(y.shape())?, &expert_out)?;
        }
        let y_reshaped = y.reshape((topk_weight.dims()[0], topk_weight.dims()[1], self.num_activated_experts, y.dims()[2]))?;
        let y_sum = topk_weight.unsqueeze(2)?.matmul(&y_reshaped)?.squeeze(2)?;
        y_sum + y_shared
    }
}

#[derive(Debug, Clone)]
pub struct HDDoubleStreamBlock {
    ada_ln_modulation: Modulation12,
    img_norm1: LayerNorm,
    img_norm2: LayerNorm,
    txt_norm1: LayerNorm,
    txt_norm2: LayerNorm,
    attn1: HDAttention,
    ff_i: HDMOEFeedForwardSwiGLU,
    ff_t: HDFeedForwardSwiGLU,
}

impl HDDoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let ada_ln_modulation = Modulation12::new(inner_dim, vb)?;
        let img_norm1 = layer_norm(inner_dim, vb.pp("norm1_i"))?;
        let img_norm2 = layer_norm(inner_dim, vb.pp("norm3_i"))?;
        let txt_norm1 = layer_norm(inner_dim, vb.pp("norm1_t"))?;
        let txt_norm2 = layer_norm(inner_dim, vb.pp("norm3_t"))?;
        let attn1 = HDAttention::new(inner_dim, cfg.num_attention_heads, cfg.attention_head_dim, false, vb.pp("attn1"))?;
        let ff_i = HDMOEFeedForwardSwiGLU::new(inner_dim, 4 * inner_dim, cfg.num_routed_experts, cfg.num_activated_experts, vb.pp("ff_i"))?;
        let ff_t = HDFeedForwardSwiGLU::new(inner_dim, 4 * inner_dim, vb.pp("ff_t"))?;
        Ok(Self {
            ada_ln_modulation,
            img_norm1,
            img_norm2,
            txt_norm1,
            txt_norm2,
            attn1,
            ff_i,
            ff_t,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let modulation = self.ada_ln_modulation.forward(vec_)?;
        let [img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate,
             txt_msa_shift, txt_msa_scale, txt_msa_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate] = modulation;

        // Attention block
        let img_norm = img.apply(&self.img_norm1)?;
        let txt_norm = txt.apply(&self.txt_norm1)?;
        
        let img_norm = img_norm.broadcast_mul(&(img_msa_scale + 1.0)?)?
            .broadcast_add(&img_msa_shift)?;
        let txt_norm = txt_norm.broadcast_mul(&(txt_msa_scale + 1.0)?)?
            .broadcast_add(&txt_msa_shift)?;

        let (img_attn, txt_attn) = self.attn1.forward_dual(&img_norm, &txt_norm, pe)?;
        
        let img = (img + img_msa_gate.broadcast_mul(&img_attn)?)?;
        let txt = (txt + txt_msa_gate.broadcast_mul(&txt_attn)?)?;

        // Feed forward block
        let img_norm = img.apply(&self.img_norm2)?;
        let txt_norm = txt.apply(&self.txt_norm2)?;
        
        let img_norm = img_norm.broadcast_mul(&(img_mlp_scale + 1.0)?)?
            .broadcast_add(&img_mlp_shift)?;
        let txt_norm = txt_norm.broadcast_mul(&(txt_mlp_scale + 1.0)?)?
            .broadcast_add(&txt_mlp_shift)?;

        let img_ff = self.ff_i.forward(&img_norm)?;
        let txt_ff = self.ff_t.forward(&txt_norm)?;
        
        let img = (img + img_mlp_gate.broadcast_mul(&img_ff)?)?;
        let txt = (txt + txt_mlp_gate.broadcast_mul(&txt_ff)?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
pub struct HDSingleStreamBlock {
    ada_ln_modulation: Modulation1,
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn1: HDAttention,
    ff_i: HDMOEFeedForwardSwiGLU,
}

impl HDSingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let ada_ln_modulation = Modulation1::new(inner_dim, vb)?;
        let norm1 = layer_norm(inner_dim, vb.pp("norm1_i"))?;
        let norm2 = layer_norm(inner_dim, vb.pp("norm3_i"))?;
        let attn1 = HDAttention::new(inner_dim, cfg.num_attention_heads, cfg.attention_head_dim, true, vb.pp("attn1"))?;
        let ff_i = HDMOEFeedForwardSwiGLU::new(inner_dim, 4 * inner_dim, cfg.num_routed_experts, cfg.num_activated_experts, vb.pp("ff_i"))?;
        Ok(Self {
            ada_ln_modulation,
            norm1,
            norm2,
            attn1,
            ff_i,
        })
    }

    fn forward(&self, x: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln_modulation.forward(vec_)?;
        
        // Attention block
        let x_norm = x.apply(&self.norm1)?;
        let x_norm = modulation.scale_shift(&x_norm)?;
        let x_attn = self.attn1.forward_single(&x_norm, pe)?;
        let x = (x + modulation.gate(&x_attn)?)?;

        // Feed forward block
        let x_norm = x.apply(&self.norm2)?;
        let x_norm = modulation.scale_shift(&x_norm)?;
        let x_ff = self.ff_i.forward(&x_norm)?;
        let x = (x + modulation.gate(&x_ff)?)?;

        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct HDLastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl HDLastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear_ = linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?;
        let ada_ln_modulation = linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear: linear_,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
pub struct TextProjection {
    linear: Linear,
}

impl TextProjection {
    fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(in_features, hidden_size, vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl candle::Module for TextProjection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

// Trait for models with forward method
pub trait WithForward {
    #[allow(clippy::too_many_arguments)]
    fn forward_with_cfg(
        &self,
        hidden_states: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &[Tensor],
        pooled_embeds: &Tensor,
        img_sizes: Option<&Tensor>,
        img_ids: Option<&Tensor>,
        llama_layers: &[usize],
    ) -> Result<Tensor>;
}

#[derive(Debug, Clone)]
pub struct HDQuantizedModel {
    x_embedder: Linear,
    t_embedder: MlpEmbedder,
    p_embedder: MlpEmbedder,
    pe_embedder: EmbedNd,
    double_blocks: Vec<HDDoubleStreamBlock>,
    single_blocks: Vec<HDSingleStreamBlock>,
    final_layer: HDLastLayer,
    caption_projection: Vec<TextProjection>,
    max_seq: usize,
}

impl HDQuantizedModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let x_embedder = linear(cfg.in_channels, inner_dim, vb.pp("x_embedder.proj"))?;
        let t_embedder = MlpEmbedder::new(256, inner_dim, vb.pp("t_embedder"))?;
        let p_embedder = MlpEmbedder::new(cfg.text_emb_dim, inner_dim, vb.pp("p_embedder"))?;
        
        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_d = vb.pp("double_stream_blocks");
        for idx in 0..cfg.num_layers {
            let db = HDDoubleStreamBlock::new(cfg, vb_d.pp(idx).pp("block"))?;
            double_blocks.push(db)
        }
        
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_s = vb.pp("single_stream_blocks");
        for idx in 0..cfg.num_single_layers {
            let sb = HDSingleStreamBlock::new(cfg, vb_s.pp(idx).pp("block"))?;
            single_blocks.push(sb)
        }
        
        let final_layer = HDLastLayer::new(inner_dim, cfg.patch_size, cfg.out_channels, vb.pp("final_layer"))?;
        
        // Create caption projection layers as per Python reference
        // Should have (num_layers + num_single_layers + 1) projection layers
        // First (num_layers + num_single_layers) layers project LLaMA embeddings (4096 -> inner_dim)
        // Last layer projects T5 embeddings (text_emb_dim -> inner_dim)
        let mut caption_projection = Vec::new();
        
        // LLaMA projection layers for each double and single block
        for i in 0..(cfg.num_layers + cfg.num_single_layers) {
            let proj = TextProjection::new(4096, inner_dim, vb.pp(&format!("caption_projection.{}", i)))?;
            caption_projection.push(proj);
        }
        
        // T5 projection layer (last one)
        let t5_proj = TextProjection::new(cfg.text_emb_dim, inner_dim, vb.pp(&format!("caption_projection.{}", cfg.num_layers + cfg.num_single_layers)))?;
        caption_projection.push(t5_proj);
        
        let pe_embedder = EmbedNd::new(10000, vec![cfg.axes_dims_rope.0, cfg.axes_dims_rope.1]);
        let max_seq = cfg.max_resolution.0 * cfg.max_resolution.1 / (cfg.patch_size * cfg.patch_size);
        
        Ok(Self {
            x_embedder,
            t_embedder,
            p_embedder,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
            caption_projection,
            max_seq,
        })
    }
}

impl WithForward for HDQuantizedModel {
    fn forward_with_cfg(
        &self,
        hidden_states: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &[Tensor], // [t5_embeds, llama_embeds]
        pooled_embeds: &Tensor,
        _img_sizes: Option<&Tensor>,
        img_ids: Option<&Tensor>,
        llama_layers: &[usize], // Which LLaMA layers to use
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        // Store original sequence length for later use
        let img_seq_len = seq_len;
        
        // Embed patches
        let embedded_states = self.x_embedder.forward(hidden_states)?;
        
        // Timestep embedding
        let timestep_emb = timestep_embedding(timesteps, 256, embedded_states.dtype())?.apply(&self.t_embedder)?;
        
        // Pooled embedding
        let pooled_emb = self.p_embedder.forward(pooled_embeds)?;
        
        // Combine timestep and pooled embeddings
        let vec = (timestep_emb + pooled_emb)?;
        
        // Positional encoding
        let pe = if let Some(ids) = img_ids {
            self.pe_embedder.forward(ids)?
        } else {
            // Default positional encoding for square images
            let h = (seq_len as f64).sqrt() as usize;
            let w = h;
            let mut ids = Vec::new();
            for i in 0..h {
                for j in 0..w {
                    ids.push([0.0, i as f32, j as f32]);
                }
            }
            let ids_tensor = Tensor::from_vec(
                ids.into_iter().flatten().collect::<Vec<f32>>(),
                (1, seq_len, 3),
                embedded_states.device(),
            )?;
            self.pe_embedder.forward(&ids_tensor)?
        };
        
        // Process text embeddings using prepare_contexts
        let t5_embeds = &encoder_hidden_states[0];
        let llama_embeds = if encoder_hidden_states.len() > 1 {
            &encoder_hidden_states[1]
        } else {
            // Create dummy LLaMA embeddings if not provided
            let (_, t5_seq_len, _) = t5_embeds.dims3()?;
            &Tensor::zeros((batch_size, llama_layers.len(), t5_seq_len, 4096), t5_embeds.dtype(), t5_embeds.device())?
        };
        
        // Prepare contexts using caption projection layers
        let contexts = self.prepare_contexts(llama_embeds, t5_embeds, llama_layers)?;
        
        // Initialize text embeddings for double stream
        // Python: txt_init = torch.cat([contexts[-1], contexts[-2]], dim=-2)
        let t5_context = &contexts[contexts.len() - 1]; // T5 (last)
        let llama_last_context = &contexts[contexts.len() - 2]; // Last LLaMA layer
        let mut txt_init = Tensor::cat(&[t5_context, llama_last_context], 1)?;
        let txt_init_len = txt_init.dim(1)?;
        
        let mut img = embedded_states;
        
        // Double stream blocks
        for (block_idx, block) in self.double_blocks.iter().enumerate() {
            // Get LLaMA context for this block
            let txt_llama = &contexts[block_idx];
            
            // Concatenate: txt_init + txt_llama (current block's LLaMA embedding)
            let txt = Tensor::cat(&[&txt_init, txt_llama], 1)?;
            
            let (new_img, new_txt_init) = block.forward(&img, &txt, &vec, &pe)?;
            img = new_img;
            
            // Extract only the txt_init part (first txt_init_len tokens)
            txt_init = new_txt_init.narrow(1, 0, txt_init_len)?;
        }
        
        // Concatenate img and txt_init for single stream
        let mut x = Tensor::cat(&[img, txt_init], 1)?;
        let joint_len = x.dim(1)?;
        
        // Single stream blocks
        for (block_idx, block) in self.single_blocks.iter().enumerate() {
            // Get LLaMA context for this single block (offset by num_layers)
            let txt_llama = &contexts[self.double_blocks.len() + block_idx];
            
            // Concatenate: x + txt_llama
            x = Tensor::cat(&[&x, txt_llama], 1)?;
            
            x = block.forward(&x, &vec, &pe)?;
            
            // Slice off the txt_llama part, keep only joint_len
            x = x.narrow(1, 0, joint_len)?;
        }
        
        // Extract image part for final layer
        let final_img = x.narrow(1, 0, img_seq_len)?;
        
        // Final layer
        let output = self.final_layer.forward(&final_img, &vec)?;
        
        Ok(output)
    }
}

impl HDQuantizedModel {
    /// Prepare contexts from LLaMA and T5 embeddings using caption projection layers
    /// This matches the Python reference prepare_contexts method
    fn prepare_contexts(
        &self,
        llama_embeds: &Tensor,  // Shape: [batch, num_llama_layers, seq_len, 4096]
        t5_embeds: &Tensor,     // Shape: [batch, seq_len, text_emb_dim]
        llama_layers: &[usize], // Which LLaMA layers to use
    ) -> Result<Vec<Tensor>> {
        let (batch_size, _, seq_len, _) = llama_embeds.dims4()?;
        
        // Extract specific LLaMA layers as per Python: contexts = [contexts[k] for k in self.llama_layers]
        let mut contexts = Vec::new();
        
        // Process LLaMA embeddings through caption projection layers
        for (i, &layer_idx) in llama_layers.iter().enumerate() {
            let llama_layer = llama_embeds.i((.., layer_idx, .., ..))?; // [batch, seq_len, 4096]
            let projected = self.caption_projection[i].forward(&llama_layer)?; // [batch, seq_len, inner_dim]
            contexts.push(projected);
        }
        
        // Process T5 embeddings through the last caption projection layer
        let t5_projected = self.caption_projection.last().unwrap().forward(t5_embeds)?;
        contexts.push(t5_projected);
        
        Ok(contexts)
    }
}
