from core.models.quantized_mobileViT import QMobileViT
import torch
import torch.nn as nn
from core.models import *
def test_mobilevit():
    model = QMobileViT(
        dim= [144, 192, 240],
        depth=[2, 4, 3],
        channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        expansion=4,
        num_classes=10,
        image_height=32,
        image_width=32,
    )
    print(model)
    return model

input_tensor = torch.randn(2, 3, 32, 32).to("cuda")
model = test_mobilevit().to("cuda")
output = model(input_tensor)

# Define a dummy loss function and compute a backward pass for testing
target = torch.randn_like(output)  # Generating a target tensor of the same shape as output
criterion = nn.MSELoss()  # Using Mean Squared Error loss for testing
loss = criterion(output, target)

# Backward pass
loss.backward()
print("Backward pass completed")