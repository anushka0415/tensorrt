import torch
import torch_tensorrt
from torch_tensorrt import Input
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import json
import argparse
import torchvision.models as models
from metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    specificity,
)

def load_and_convert_model(model_path, input_shape, precision=torch.half):
    resnet18 = models.resnet18(pretrained=True).half().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]

    trt_model = torch_tensorrt.compile(
        resnet18,
        inputs=inputs,
        enabled_precisions={precision}
    )

    return trt_model

def inference(trt_model, img, topk=5):
    trt_model.eval()
    with torch.no_grad():
        outputs = trt_model(img)

    prob = torch.nn.functional.softmax(outputs[0], dim=1)
    probs, classes = torch.topk(prob, topk, dim=1)

    return probs, classes


def preprocess_image(img_dir, img_path):
    img_path = os.path.join(img_dir, img_path)
    image = Image.open(img_path).convert("RGB")
    input_image = image.resize((224, 224))
    input_image_array = np.array(input_image)
    img = transforms.ToTensor()(input_image_array).unsqueeze(0).to("cuda")
    return img

def evaluate_images_tensorrt(trt_model, split_path, labels):
    trt_model.eval()
    
    true_labels = []
    predicted_labels = []

    for class_folder in os.listdir(split_path):
        class_path = os.path.join(split_path, class_folder)

        for img_path in os.listdir(class_path):
            input_img = preprocess_image(class_path, img_path).half()
            print(f"Input image shape: {input_img.shape}") 
            probs, classes = inference(trt_model, input_img, topk=1)

            predicted_idx = int(classes[0])
            predicted_label = labels[predicted_idx]
            print(predicted_idx, predicted_label)
            predicted_labels.append(predicted_idx)
            
            true_label = class_folder 
            true_labels.append(true_label)
            true_labels=[564,756] #just dummy values 

    numeric_actual_labels = np.array(true_labels)
    numeric_predicted_labels = np.array(predicted_labels).reshape(1, -1)
    print(numeric_predicted_labels.shape)

    true_labels = torch.tensor(numeric_actual_labels).view(1, -1)[:, 0].view(-1)
    predicted_labels = torch.tensor(numeric_predicted_labels)

    metrics = {}
    metrics['acc1'] = accuracy(predicted_labels, true_labels, topk=(1,))[0]
    metrics['precision'] = precision(predicted_labels, true_labels)
    metrics['recall'] = recall(predicted_labels, true_labels)
    metrics['f1_score'] = f1_score(predicted_labels, true_labels)
    metrics['specificity'] = specificity(predicted_labels, true_labels)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

def main():
    # Set paths relative to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "resnet18.pt")
    split_path = os.path.join(current_dir, "images/test")
    categories_path = os.path.join(current_dir, "imagenet.json")

    # Load and convert the model
    input_shape = (1, 3, 224, 224)
    trt_model = load_and_convert_model(model_path, input_shape)

    # Load labels
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    # Evaluate images
    evaluate_images_tensorrt(trt_model, split_path, categories)

if __name__ == "__main__":
    main()
