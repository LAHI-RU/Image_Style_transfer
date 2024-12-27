from PIL import Image
import torch
from torchvision import transforms

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for the VGG-19 model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(400),  # Resize the image to 400 pixels on the longest side
        transforms.CenterCrop(400),  # Crop to 400x400 to ensure a consistent size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply the preprocessing transformations
    image_tensor = preprocess(image)

    # Add a batch dimension (required for PyTorch models)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
