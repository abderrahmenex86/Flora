import json
import os

import torch
from PIL import Image
from torchvision.transforms import v2

from src.factory import create_model_instance


def execute_inference(run_directory_path, input_image_path):
    hyperparameters_file_path = os.path.join(run_directory_path, "hyperparameters.json")
    best_model_file_path = os.path.join(run_directory_path, "best_model.pth")
    class_names_file_path = os.path.join(run_directory_path, "class_names.json")

    with open(hyperparameters_file_path, "r") as json_file_pointer:
        hyperparameters = json.load(json_file_pointer)

    num_classes = hyperparameters["num_classes"]
    device_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_instance = create_model_instance(num_classes=num_classes)
    model_instance.load_state_dict(torch.load(best_model_file_path, map_location=device_target))
    model_instance.to(device_target)
    model_instance.eval()

    inference_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    raw_image = Image.open(input_image_path).convert("RGB")
    transformed_image_tensor = inference_transform(raw_image).unsqueeze(0).to(device_target)

    with torch.no_grad():
        model_predictions = model_instance(transformed_image_tensor)
        predicted_class_index = torch.argmax(model_predictions, dim=-1).item()

    class_names_list = None
    if os.path.exists(class_names_file_path):
        with open(class_names_file_path, "r") as json_file_pointer:
            class_names_list = json.load(json_file_pointer)

    if class_names_list:
        predicted_class_label = class_names_list[predicted_class_index]
        print(f"Predicted class index: {predicted_class_index} | Label: {predicted_class_label}")
    else:
        print(f"Predicted class index: {predicted_class_index}")
