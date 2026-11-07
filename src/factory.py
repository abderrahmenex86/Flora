import torch
import torch.nn as nn

from src.models import MulticlassFocalLoss, PlantClassifier


def create_model_instance(num_classes):
    return PlantClassifier(num_classes=num_classes)


def create_loss_criterion(loss_type="cross_entropy", label_smoothing_value=0.01, focal_gamma_value=2.0):
    if loss_type == "focal":
        return MulticlassFocalLoss(gamma_value=focal_gamma_value)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing_value)


def create_stratified_optimizer(model_instance, base_learning_rate=1e-3, weight_decay_value=1e-1):
    encoder_parameters = model_instance.backbone_network.features.parameters()
    classifier_parameters = model_instance.backbone_network.classifier.parameters()

    parameter_groups = [
        {"params": encoder_parameters, "lr": base_learning_rate * 0.01},
        {"params": classifier_parameters, "lr": base_learning_rate},
    ]

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay_value)


def create_learning_rate_scheduler(optimizer_instance, factor_value=0.5, patience_value=4):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_instance, mode="max", factor=factor_value, patience=patience_value
    )
