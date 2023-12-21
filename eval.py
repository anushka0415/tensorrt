import torch
from torch_tensorrt import torch2trt, Input
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import json

def load_and_convert_model(model_path, input_shape, precision=torch.float32):
    # Load PyTorch model
    model = torch.load(model_path)

    # Trace the model
    traced_model = torch.jit.trace(model, torch.randn(input_shape).to("cuda"))

    # Convert to TensorRT format
    trt_model = torch2trt(
        traced_model,
        inputs=[Input(input_shape, dtype=precision)],
        enabled_precisions={precision}
    )

    return trt_model

def inference(trt_model, img_batch, topk=5):
    trt_model.eval()
    with torch.no_grad():
        outputs = trt_model(img_batch)
    prob = torch.nn.functional.softmax(outputs[0], dim=1)

    probs, classes = torch.topk(prob, topk, dim=1)
    return probs, classes

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_image = image.resize((224, 224))
    input_image_array = np.array(input_image)
    img = transforms.ToTensor()(img).unsqueeze(0).to("cuda")
    return img

def main(model_path, img_dir):
    # Load and convert the model
    input_shape = (1, 3, 224, 224)  # Adjust the input shape based on your model
    trt_model = load_and_convert_model(model_path, input_shape)

    # Load the batch of images
    img_batch = torch.cat([preprocess_image(img_path) for img_path in os.listdir(img_dir)])

    # Run inference
    topk = 5  # Number of top predictions
    probs, classes = inference(trt_model, img_batch, topk)
    
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    # Display results
    # Replace with your class labels
    for i in os.listdir(img_dir):
        print(f"Results for {i}:")
        for j in range(topk):
            probability = probs[i][j].item()
            class_label = categories[int(classes[i][j])]
            print(f"Top {j + 1}: {class_label} - Probability: {probability * 100:.2f}%")

if __name__ == "__main__":
    model_path = ""  # Replace with your model path
    img_dir = "path/to/your/image_folder"  # Replace with your image paths
    categories_path='path to json with categories'
    main(model_path, img_dir)
