from core.models.quantized_mobileViT import QMobileViT
import torch
import torch.nn as nn
from core.models import *
def test_mobilevit():
    model = QMobileViT()
    # print(model)
    return model

input_tensor = torch.randn(1, 3, 256, 256).to("cuda")
model = test_mobilevit().to("cuda")
model.set_input_bitwidth(8)
model.set_weight_bitwidth(8)
model.set_output_bitwidth(8)
# print(model)
# exit(0)
output = model(input_tensor)
print(model)
exit(0)

# Define a dummy loss function and compute a backward pass for testing
target = torch.randn_like(output)  # Generating a target tensor of the same shape as output
criterion = nn.MSELoss()  # Using Mean Squared Error loss for testing
loss = criterion(output, target)

# Backward pass
loss.backward()
print("Backward pass completed")

for param in model.parameters():

    print(param.shape)  # Prints the shape of each parameter