"""Minimal, dependency-free LoRA support for linear layers."""

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to an existing linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_output + lora_output * self.scaling


def parse_lora_targets(targets: str) -> Sequence[str]:
    parsed = tuple(target.strip() for target in targets.split(",") if target.strip())
    if not parsed:
        raise ValueError("--lora_target must contain at least one module name.")
    return parsed


def inject_lora(
    model: nn.Module,
    targets: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Replace matching linear modules and return their fully-qualified names."""
    target_names = set(targets)
    replacements = []
    for module_name, module in list(model.named_modules()):
        if not module_name or isinstance(module, LoRALinear):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name in target_names and isinstance(module, nn.Linear):
            replacements.append((module_name, module))

    replaced_names = []
    for module_name, module in replacements:
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced_names.append(module_name)
    return replaced_names


def find_lora_module_ranks(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Infer LoRA module paths and ranks from checkpoint tensor names/shapes."""
    suffix = ".lora_A"
    modules = {}
    for key, tensor in state_dict.items():
        if key.endswith(suffix):
            if tensor.ndim != 2:
                raise ValueError(f"Invalid LoRA tensor '{key}': expected 2 dimensions.")
            module_name = key[:-len(suffix)]
            b_key = f"{module_name}.lora_B"
            if b_key not in state_dict:
                raise ValueError(f"LoRA checkpoint is missing matching tensor '{b_key}'.")
            modules[module_name] = tensor.shape[0]
    return modules


def inject_lora_from_state_dict(
    model: nn.Module,
    module_ranks: Mapping[str, int],
    alpha: float,
    dropout: float,
) -> List[str]:
    """Inject LoRA at exact module paths inferred from a checkpoint."""
    replaced_names = []
    for module_name, rank in module_ranks.items():
        try:
            module = model.get_submodule(module_name)
        except AttributeError as error:
            raise ValueError(
                f"LoRA checkpoint module '{module_name}' does not exist in the selected model."
            ) from error
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoRA checkpoint target '{module_name}' must be nn.Linear, "
                f"but found {type(module).__name__}."
            )
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced_names.append(module_name)
    return replaced_names


def mark_only_lora_and_head_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    for name, parameter in model.named_parameters():
        if name == "head" or name.startswith("head."):
            parameter.requires_grad = True
