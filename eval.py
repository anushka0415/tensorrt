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

def load_and_convert_model(model_path, input_shape, precision=torch.half):
    # Load PyTorch model
    #model = torch.load(model_path).to("cuda")

    # Trace the model
    resnet18 = models.resnet18(pretrained=True).half().to("cuda")
    #traced_model = torch.jit.trace(resnet18, torch.randn(input_shape).to("cuda"))
   
    #traced_model=torch.jit.load(model_path).to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]
    # Convert to TensorRT format
    trt_model = torch_tensorrt.compile(
        resnet18,
        inputs=inputs,
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

def preprocess_image(img_dir,img_path):
    img_path=os.path.join(img_dir, img_path)
    image = Image.open(img_path).convert("RGB")
    input_image = image.resize((224, 224))
    input_image_array = np.array(input_image)
    img = transforms.ToTensor()(input_image_array).unsqueeze(0).to("cuda")
    return img

def main(model_path, img_dir,categories_path):
    # Load and convert the model
    input_shape = (1, 3, 224, 224)  # Adjust the input shape based on your model
    trt_model = load_and_convert_model(model_path, input_shape)

    # Load the batch of images
    img_batch = torch.cat([preprocess_image(img_dir,img_path).half() for img_path in os.listdir(img_dir)])

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
            probability = probs[j].item()
            class_index = int(classes[i][j])
        
        # Get label from 'categories' dictionary
            if str(class_index) in categories:
                class_label = categories[str(class_index)][1]
            else:
                class_label = "Unknown"
        print(f"Top {j + 1}: {class_label} - Probability: {probability * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--img_dir", type=str, help="Path to the image folder")
    parser.add_argument("--categories_path", type=str, help="Path to the JSON file with categories")

    args = parser.parse_args()

    main(args.model_path, args.img_dir, args.categories_path)
