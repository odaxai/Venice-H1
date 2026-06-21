"""
LoRA (Low-Rank Adaptation) for BEiT-3 Backbone
================================================
Manual LoRA implementation for torchscale's MultiwayNetwork-wrapped
attention layers. Enables medical domain adaptation without ruining
natural image performance.

BEiT-3 uses MultiwayNetwork which wraps each nn.Linear projection
into .A (vision path) and .B (text path). We apply LoRA to both
paths in the last N encoder layers.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
Models", ICLR 2022.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wraps an existing frozen nn.Linear with a trainable low-rank adapter.

    output = frozen_linear(x) + lora_B(lora_A(x)) * scaling

    The original linear weights are kept frozen. Only lora_A and lora_B
    are trained. lora_B is initialized to zeros so at init the output
    is identical to the original frozen linear.
    """

    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # LoRA decomposition: W' = W + B @ A * scaling
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (so delta=0 at init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, **kwargs):
        # Original frozen path
        original_out = self.original_linear(x, **kwargs)

        # LoRA path (trainable)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        return original_out + lora_out

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, "
                f"scaling={self.scaling:.3f}")


def apply_lora_to_model(model, rank=8, alpha=16, num_lora_layers=6,
                        dropout=0.05, target_projections=('q_proj', 'k_proj', 'v_proj')):
    """
    Apply LoRA adapters to the BEiT-3 backbone's attention layers.

    Targets the last `num_lora_layers` encoder layers, wrapping the
    specified projection modules in both the .A (vision) and .B (text)
    multiway paths.

    Args:
        model: GridCellOneRef model instance
        rank: LoRA rank (default: 8)
        alpha: LoRA scaling alpha (default: 16)
        num_lora_layers: Number of layers from the end to apply LoRA to
        dropout: Dropout on LoRA path
        target_projections: Which attention projections to target

    Returns:
        int: Total number of LoRA parameters added
    """
    encoder_layers = model.backbone.beit3.encoder.layers
    total_layers = len(encoder_layers)
    start_layer = max(0, total_layers - num_lora_layers)

    total_lora_params = 0
    adapters_applied = 0

    for layer_idx in range(start_layer, total_layers):
        layer = encoder_layers[layer_idx]
        attn = layer.self_attn

        for proj_name in target_projections:
            proj_module = getattr(attn, proj_name)

            # Check if it's a MultiwayNetwork (has .A and .B)
            if hasattr(proj_module, 'A') and hasattr(proj_module, 'B'):
                # Wrap both vision (.A) and text (.B) paths
                for path_name in ('A', 'B'):
                    original_linear = getattr(proj_module, path_name)
                    if isinstance(original_linear, nn.Linear):
                        lora_wrapped = LoRALinear(
                            original_linear, rank=rank,
                            alpha=alpha, dropout=dropout
                        )
                        setattr(proj_module, path_name, lora_wrapped)
                        n_params = sum(
                            p.numel() for p in lora_wrapped.parameters()
                            if p.requires_grad
                        )
                        total_lora_params += n_params
                        adapters_applied += 1
            elif isinstance(proj_module, nn.Linear):
                # Direct Linear (no multiway) — fallback
                lora_wrapped = LoRALinear(
                    proj_module, rank=rank,
                    alpha=alpha, dropout=dropout
                )
                setattr(attn, proj_name, lora_wrapped)
                n_params = sum(
                    p.numel() for p in lora_wrapped.parameters()
                    if p.requires_grad
                )
                total_lora_params += n_params
                adapters_applied += 1

    print(f"[LoRA] Applied {adapters_applied} adapters "
          f"(layers {start_layer}-{total_layers - 1})")
    print(f"[LoRA] Target projections: {target_projections}")
    print(f"[LoRA] Rank={rank}, Alpha={alpha}, Dropout={dropout}")
    print(f"[LoRA] Total LoRA params: {total_lora_params:,}")

    return total_lora_params


def get_lora_params(model):
    """
    Collect all LoRA parameters from the model for the optimizer.

    Returns:
        list: List of LoRA parameter tensors with requires_grad=True
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model):
    """
    Extract only LoRA weights for saving (lightweight checkpoint).

    Returns:
        dict: State dict containing only LoRA parameters
    """
    return {
        name: param.data
        for name, param in model.named_parameters()
        if 'lora_' in name
    }


def load_lora_state_dict(model, state_dict, strict=False):
    """
    Load LoRA weights from a checkpoint.

    Args:
        model: The model to load into
        state_dict: Dict of LoRA parameter name -> tensor
        strict: Whether to require all keys to match
    """
    model_dict = model.state_dict()
    lora_keys = {k for k in model_dict if 'lora_' in k}

    loaded = 0
    for key, value in state_dict.items():
        if key in model_dict:
            model_dict[key] = value
            loaded += 1
        elif strict:
            raise KeyError(f"LoRA key not found in model: {key}")

    model.load_state_dict(model_dict, strict=False)
    print(f"[LoRA] Loaded {loaded}/{len(state_dict)} LoRA parameters")
    if lora_keys:
        missing = lora_keys - set(state_dict.keys())
        if missing:
            print(f"[LoRA] {len(missing)} LoRA params not in checkpoint "
                  f"(initialized fresh)")
    return loaded
