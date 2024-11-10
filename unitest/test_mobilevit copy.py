from core.models.quantized_mobileViT import QMobileViT
import torch
import torch.nn as nn
from core.models import *
def test_mobilevit():
    model = QMobileViT()
    print(model)
    return model

input_tensor = torch.randn(1, 3, 256, 256)
model = test_mobilevit()
output = model(input_tensor)

# Define a dummy loss function and compute a backward pass for testing
target = torch.randn_like(output)  # Generating a target tensor of the same shape as output
criterion = nn.MSELoss()  # Using Mean Squared Error loss for testing
loss = criterion(output, target)

# Backward pass
loss.backward()
print("Backward pass completed")