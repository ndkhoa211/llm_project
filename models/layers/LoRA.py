import math
import torch

class LoRALayer(torch.nn.Module):
  def __init__(self, in_dim, out_dim, rank, alpha):
    super().__init__()
    self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
    # Kaiming/He uniform initialization, similar to standard weight initialization
    torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
    self.alpha = alpha

  def forward(self, x):
    x = self.alpha * (x @ self.A @ self.B)
    return x


class LinearLayerWithLoRA(torch.nn.Module):
  def __init__(self, linear, rank, alpha):
    super().__init__()
    self.linear = linear
    self.lora = LoRALayer(linear.in_features,
                          linear.out_features,
                          rank,
                          alpha)

  def forward(self, x):
    return self.linear(x) + self.lora(x)



def replace_linear_with_lora(model, rank, alpha):
  for name, module in model.named_children():
    if isinstance(module, torch.nn.Linear):
      print(f"Replacing {name} with LinearLayerWithLoRA")
      setattr(model, name, LinearLayerWithLoRA(module, rank, alpha))
    else:
      # recursively apply the same function to child modules
      replace_linear_with_lora(module, rank, alpha)