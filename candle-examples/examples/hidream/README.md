
📦 Loading t_embedder...
Loading MlpEmbedder: in_sz=256, h_sz=2560
  ⚠ in_layer bias not found, trying without bias: cannot find tensor t_embedder.in_layer.bias
Error: cannot find tensor t_embedder.in_layer.weight



we are working on implementing hidream quantized gguf in rust. 
@/candle-examples/examples/hidream/main.rs 
and 
@/candle-transformers/src/models/hidream/quantized_model.rs

we have code in flux that uses gguf. reference that to add support for hidream.
@/candle-examples/examples/flux/main.rs is an example
@/candle-transformers/src/models/flux/quantized_model.rs 

and our hideam gguf models are here
https://huggingface.co/ND911/hidream_i1_fp8_full_dev_fast_ggufs


# Reference code
our code:
@candle-examples/examples/hidream/main.rs 
@candle-transformers/src/models/hidream/mod.rs 

our reference:
@candle-examples/examples/hidream/reference/pipeline_hidream_image.py
and
@candle-transformers/src/models/hidream/reference/model.py

flux has many similar systems. 
check how they do things. 
@candle-examples/examples/flux/
@candle-transformers/src/models/flux 


refernce the python