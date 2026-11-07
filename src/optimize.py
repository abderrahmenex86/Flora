import optuna
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm.auto import tqdm

from src.factory import (
    create_loss_criterion,
    create_model_instance,
    create_stratified_optimizer,
)
from src.trainer import run_evaluation_epoch, run_training_epoch

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_study(train_dataset, validation_dataset, num_classes, total_trials_count=10, epochs_per_trial=10):
    subset_train_indices = list(range(min(512, len(train_dataset))))
    subset_validation_indices = list(range(min(128, len(validation_dataset))))

    subset_train_dataset = Subset(train_dataset, subset_train_indices)
    subset_validation_dataset = Subset(validation_dataset, subset_validation_indices)

    train_loader = DataLoader(subset_train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(subset_validation_dataset, batch_size=32, shuffle=False)

    device_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective_function(trial):
        suggested_learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        suggested_weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
        suggested_loss_type = trial.suggest_categorical("loss_type", ["cross_entropy", "focal"])

        model_instance = create_model_instance(num_classes=num_classes).to(device_target)
        optimizer_instance = create_stratified_optimizer(
            model_instance, base_learning_rate=suggested_learning_rate, weight_decay_value=suggested_weight_decay
        )
        loss_criterion = create_loss_criterion(loss_type=suggested_loss_type)

        metric_top1 = MulticlassAccuracy(num_classes=num_classes).to(device_target)
        metric_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device_target)
        metric_f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device_target)

        trial_best_f1 = 0.0

        for epoch_index in range(epochs_per_trial):
            run_training_epoch(
                model_instance=model_instance,
                dataloader_instance=train_loader,
                loss_criterion=loss_criterion,
                optimizer_instance=optimizer_instance,
                device=device_target,
                metric_top1=metric_top1,
                metric_top5=metric_top5,
                metric_f1=metric_f1,
            )
            validation_metrics = run_evaluation_epoch(
                model_instance=model_instance,
                dataloader_instance=validation_loader,
                loss_criterion=loss_criterion,
                device=device_target,
                metric_top1=metric_top1,
                metric_top5=metric_top5,
                metric_f1=metric_f1,
            )
            trial_best_f1 = max(trial_best_f1, validation_metrics["f1_score"])

        return trial_best_f1

    optuna_study = optuna.create_study(direction="maximize")
    optimization_progress_bar = tqdm(range(total_trials_count), desc="Optimization Search")

    for _ in optimization_progress_bar:
        trial_instance = optuna_study.ask()
        trial_value = objective_function(trial_instance)
        optuna_study.tell(trial_instance, trial_value)
        optimization_progress_bar.set_postfix({"best_f1_score": f"{optuna_study.best_value:.2f}%"})

    best_hyperparameters = optuna_study.best_params
    rounded_learning_rate = round(best_hyperparameters["learning_rate"], 6)
    rounded_weight_decay = round(best_hyperparameters["weight_decay"], 4)
    best_loss_type = best_hyperparameters["loss_type"]

    print("\nOptimization study completed. Copy and run the following command to start training:")
    print(
        f"python main.py --mode train --learning_rate {rounded_learning_rate} --weight_decay {rounded_weight_decay} --loss_type {best_loss_type}\n"
    )

    return best_hyperparameters
