import torch
from torch.utils.data import DataLoader, Subset

from src.factory import (
    create_loss_criterion,
    create_model_instance,
    create_stratified_optimizer,
)


def perform_sanity_check(train_dataset, num_classes, base_learning_rate):
    toy_indices = list(range(5))
    toy_dataset = Subset(train_dataset, toy_indices)
    toy_dataloader = DataLoader(toy_dataset, batch_size=5, shuffle=False)

    input_tensors, target_labels = next(iter(toy_dataloader))

    assert input_tensors.shape == torch.Size([5, 3, 224, 224]), f"Unexpected input shape: {input_tensors.shape}"
    assert target_labels.shape == torch.Size([5]), f"Unexpected target shape: {target_labels.shape}"

    model_instance = create_model_instance(num_classes=num_classes)
    optimizer_instance = create_stratified_optimizer(model_instance, base_learning_rate=base_learning_rate)
    loss_criterion = create_loss_criterion()

    model_outputs = model_instance(input_tensors)
    assert model_outputs.shape == torch.Size([5, num_classes]), f"Unexpected output shape: {model_outputs.shape}"

    loss_value = loss_criterion(model_outputs, target_labels)
    loss_value.backward()

    for parameter_name, parameter_tensor in model_instance.named_parameters():
        if parameter_tensor.requires_grad:
            assert parameter_tensor.grad is not None, f"Gradient not found for parameter: {parameter_name}"

    optimizer_groups = optimizer_instance.param_groups
    assert len(optimizer_groups) == 2, f"Expected 2 parameter groups, found {len(optimizer_groups)}"
    assert optimizer_groups[0]["lr"] == base_learning_rate * 0.01, f"Expected backbone learning rate to be scaled down"
    assert optimizer_groups[1]["lr"] == base_learning_rate, f"Expected classifier learning rate to be unscaled"

    print("Sanity checks passed successfully!")
