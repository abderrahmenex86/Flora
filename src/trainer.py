import datetime
import json
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm.auto import tqdm

from src.factory import (
    create_learning_rate_scheduler,
    create_loss_criterion,
    create_model_instance,
    create_stratified_optimizer,
)


def run_training_epoch(
    model_instance,
    dataloader_instance,
    loss_criterion,
    optimizer_instance,
    device,
    metric_top1,
    metric_top5,
    metric_f1,
    gradient_clip_limit=1.0,
):
    model_instance.train()
    metric_top1.reset()
    metric_top5.reset()
    metric_f1.reset()

    running_loss_sum = 0.0
    processed_samples_count = 0

    for input_tensors, target_labels in tqdm(dataloader_instance, desc="Training Batches", leave=False):
        input_tensors = input_tensors.to(device, non_blocking=True)
        target_labels = target_labels.to(device, non_blocking=True)

        optimizer_instance.zero_grad()
        model_outputs = model_instance(input_tensors)
        loss_value = loss_criterion(model_outputs, target_labels)
        loss_value.backward()

        clip_grad_norm_(model_instance.parameters(), max_norm=gradient_clip_limit)
        optimizer_instance.step()

        batch_size = input_tensors.size(0)
        processed_samples_count += batch_size
        running_loss_sum += loss_value.item() * batch_size

        metric_top1.update(model_outputs, target_labels)
        metric_top5.update(model_outputs, target_labels)
        metric_f1.update(model_outputs, target_labels)

    epoch_loss_value = running_loss_sum / processed_samples_count
    epoch_accuracy_top1 = metric_top1.compute().item()
    epoch_accuracy_top5 = metric_top5.compute().item()
    epoch_f1_score = metric_f1.compute().item()

    return {
        "loss": epoch_loss_value,
        "accuracy_top1": epoch_accuracy_top1 * 100.0,
        "accuracy_top5": epoch_accuracy_top5 * 100.0,
        "f1_score": epoch_f1_score * 100.0,
    }


def run_evaluation_epoch(
    model_instance, dataloader_instance, loss_criterion, device, metric_top1, metric_top5, metric_f1
):
    model_instance.eval()
    metric_top1.reset()
    metric_top5.reset()
    metric_f1.reset()

    running_loss_sum = 0.0
    processed_samples_count = 0

    with torch.no_grad():
        for input_tensors, target_labels in tqdm(dataloader_instance, desc="Evaluation Batches", leave=False):
            input_tensors = input_tensors.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)

            model_outputs = model_instance(input_tensors)
            loss_value = loss_criterion(model_outputs, target_labels)

            batch_size = input_tensors.size(0)
            processed_samples_count += batch_size
            running_loss_sum += loss_value.item() * batch_size

            metric_top1.update(model_outputs, target_labels)
            metric_top5.update(model_outputs, target_labels)
            metric_f1.update(model_outputs, target_labels)

    epoch_loss_value = running_loss_sum / processed_samples_count
    epoch_accuracy_top1 = metric_top1.compute().item()
    epoch_accuracy_top5 = metric_top5.compute().item()
    epoch_f1_score = metric_f1.compute().item()

    return {
        "loss": epoch_loss_value,
        "accuracy_top1": epoch_accuracy_top1 * 100.0,
        "accuracy_top5": epoch_accuracy_top5 * 100.0,
        "f1_score": epoch_f1_score * 100.0,
    }


def execute_training_loop(
    train_dataloader,
    validation_dataloader,
    dataset_classes,
    num_epochs=40,
    learning_rate=1e-3,
    weight_decay=1e-1,
    loss_type="cross_entropy",
    resume_flag=False,
    resume_directory_path=None,
    **kwargs,
):
    device_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset_classes)

    model_instance = create_model_instance(num_classes=num_classes).to(device_target)
    optimizer_instance = create_stratified_optimizer(
        model_instance=model_instance, base_learning_rate=learning_rate, weight_decay_value=weight_decay
    )
    scheduler_instance = create_learning_rate_scheduler(optimizer_instance=optimizer_instance)
    loss_criterion = create_loss_criterion(loss_type=loss_type)

    start_epoch_index = 0
    best_validation_f1_score = 0.0
    history_tracker = {
        "train": {"loss": [], "accuracy_top1": [], "accuracy_top5": [], "f1_score": []},
        "validation": {"loss": [], "accuracy_top1": [], "accuracy_top5": [], "f1_score": []},
    }

    if resume_flag:
        if resume_directory_path is None:
            artifact_root_directory = "artifacts"
            if os.path.exists(artifact_root_directory):
                past_run_directories = sorted(
                    [
                        os.path.join(artifact_root_directory, directory)
                        for directory in os.listdir(artifact_root_directory)
                        if os.path.isdir(os.path.join(artifact_root_directory, directory))
                    ]
                )
                if past_run_directories:
                    resume_directory_path = past_run_directories[-1]

        if resume_directory_path and os.path.exists(resume_directory_path):
            checkpoint_file_path = os.path.join(resume_directory_path, "checkpoint_last.pth")
            if os.path.exists(checkpoint_file_path):
                checkpoint_data_payload = torch.load(checkpoint_file_path, map_location=device_target)
                model_instance.load_state_dict(checkpoint_data_payload["model_state_dict"])
                optimizer_instance.load_state_dict(checkpoint_data_payload["optimizer_state_dict"])
                scheduler_instance.load_state_dict(checkpoint_data_payload["scheduler_state_dict"])
                start_epoch_index = checkpoint_data_payload["epoch_index"]
                best_validation_f1_score = checkpoint_data_payload["best_validation_f1_score"]
                history_tracker = checkpoint_data_payload["history"]
                output_directory_path = resume_directory_path
                print(f"Resuming training from checkpoint: {checkpoint_file_path}")
    else:
        current_time_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory_path = f"artifacts/{current_time_string}_mobilenetv3_large"
        os.makedirs(output_directory_path, exist_ok=True)

        hyperparameters_dictionary = {
            "num_classes": num_classes,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_type": loss_type,
            **kwargs,
        }
        with open(os.path.join(output_directory_path, "hyperparameters.json"), "w") as json_file_pointer:
            json.dump(hyperparameters_dictionary, json_file_pointer, indent=4)

        with open(os.path.join(output_directory_path, "class_names.json"), "w") as json_file_pointer:
            json.dump(dataset_classes, json_file_pointer, indent=4)

        try:
            architecture_text_representation = str(summary(model_instance, input_size=(1, 3, 224, 224), verbose=0))
            with open(os.path.join(output_directory_path, "architecture.txt"), "w") as text_file_pointer:
                text_file_pointer.write(architecture_text_representation)
        except Exception:
            pass

    metric_top1 = MulticlassAccuracy(num_classes=num_classes).to(device_target)
    metric_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device_target)
    metric_f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device_target)

    for epoch_loop_index in range(start_epoch_index, num_epochs):
        epoch_display_index = epoch_loop_index + 1
        print(f"\nEpoch {epoch_display_index}/{num_epochs}")

        train_metrics = run_training_epoch(
            model_instance=model_instance,
            dataloader_instance=train_dataloader,
            loss_criterion=loss_criterion,
            optimizer_instance=optimizer_instance,
            device=device_target,
            metric_top1=metric_top1,
            metric_top5=metric_top5,
            metric_f1=metric_f1,
            gradient_clip_limit=float(kwargs.get("gradient_clip_limit", 1.0)),
        )

        validation_metrics = run_evaluation_epoch(
            model_instance=model_instance,
            dataloader_instance=validation_dataloader,
            loss_criterion=loss_criterion,
            device=device_target,
            metric_top1=metric_top1,
            metric_top5=metric_top5,
            metric_f1=metric_f1,
        )

        scheduler_instance.step(validation_metrics["f1_score"])

        current_learning_rate = optimizer_instance.param_groups[-1]["lr"]
        print(f"Current Learning Rate: {current_learning_rate:.8f}")

        for metric_key in train_metrics:
            history_tracker["train"][metric_key].append(train_metrics[metric_key])
            history_tracker["validation"][metric_key].append(validation_metrics[metric_key])

        print(
            f"Train Metrics | Loss: {train_metrics['loss']:.4f} | "
            f"Top-1 Acc: {train_metrics['accuracy_top1']:.2f}% | "
            f"Top-5 Acc: {train_metrics['accuracy_top5']:.2f}% | "
            f"Macro F1: {train_metrics['f1_score']:.2f}%"
        )
        print(
            f"Val Metrics   | Loss: {validation_metrics['loss']:.4f} | "
            f"Top-1 Acc: {validation_metrics['accuracy_top1']:.2f}% | "
            f"Top-5 Acc: {validation_metrics['accuracy_top5']:.2f}% | "
            f"Macro F1: {validation_metrics['f1_score']:.2f}%"
        )

        with open(os.path.join(output_directory_path, "model_history.json"), "w") as json_file_pointer:
            json.dump(history_tracker, json_file_pointer, indent=4)

        current_validation_f1 = validation_metrics["f1_score"]

        checkpoint_state = {
            "model_state_dict": model_instance.state_dict(),
            "optimizer_state_dict": optimizer_instance.state_dict(),
            "scheduler_state_dict": scheduler_instance.state_dict(),
            "epoch_index": epoch_display_index,
            "best_validation_f1_score": best_validation_f1_score,
            "history": history_tracker,
        }

        torch.save(checkpoint_state, os.path.join(output_directory_path, "checkpoint_last.pth"))

        if current_validation_f1 > best_validation_f1_score:
            best_validation_f1_score = current_validation_f1
            torch.save(model_instance.state_dict(), os.path.join(output_directory_path, "best_model.pth"))
            print("New best validation F1-score achieved. Best model checkpoint saved!")

    return history_tracker
