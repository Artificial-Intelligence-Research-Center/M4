from transformers import ViTForImageClassification
import torch

model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224")

torch.save(model.state_dict(), "vit_large_patch16_224.pth")