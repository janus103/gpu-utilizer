# Layer CSV Schema

The profiler uses a superset schema so different model extractors can target one format. Unknown or unused values should be `-1`.

## Required Identity Columns

- `layer_seq`: actual execution order.
- `layer_type`: supported operator type.
- `layer_name`: optional readable label.

Supported `layer_type` values:

- `standard_conv`
- `depthwise_conv`
- `pointwise_conv`
- `fully_connected`
- `matrix`
- `self_attention`

## Common Columns

- `batch`
- `dtype`: `float32`, `float16`, or `bfloat16`
- `device`
- `repeat`
- `warmup`

## Convolution Columns

- `in_channels`
- `out_channels`
- `input_h`
- `input_w`
- `kernel_h`
- `kernel_w`
- `stride_h`
- `stride_w`
- `pad_h`
- `pad_w`
- `groups`
- `bias`

## Fully Connected / Matrix Columns

- `in_features`
- `out_features`
- `m`
- `n`
- `k`

## Self Attention Columns

- `seq_len`
- `embed_dim`
- `num_heads`
- `head_dim`
- `qkv_bias`
- `causal`

`causal` is currently normalized and recorded, but the first implementation uses PyTorch `MultiheadAttention` without a custom causal mask.
