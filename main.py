from preprocess import load_and_preprocess_image
from style_transfer import perform_style_transfer

# Paths to your images
content_image_path = "content.jpg"
style_image_path = "style.jpg"
output_image_path = "output/stylized_image.jpg"

# Load and preprocess the images
content_tensor = load_and_preprocess_image(content_image_path)
style_tensor = load_and_preprocess_image(style_image_path)

print(f"Content Image Tensor Shape: {content_tensor.shape}")
print(f"Style Image Tensor Shape: {style_tensor.shape}")
