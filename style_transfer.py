import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

content_img = load_image('content.jpg')
style_img = load_image('style.jpg')

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:21]

    def forward(self, x):
        return self.model(x)

def content_loss(content_features, target_features):
    return torch.mean((content_features - target_features)**2)

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

def style_loss(style_grams, target_grams):
    loss = 0
    for sg, tg in zip(style_grams, target_grams):
        loss += torch.mean((sg - tg)**2)
    return loss


def train(content_img, style_img, num_steps=500):
    target_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target_img], lr=0.01)
    model = VGG().to(device)
    for step in range(num_steps):
        target_features = model(target_img)
        content_features = model(content_img)
        style_features = model(style_img)

        loss = content_loss(content_features, target_features) + \
               style_loss(gram_matrix(style_features), gram_matrix(target_features))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return target_img

output = train(content_img, style_img)
output_image = output.squeeze().detach()
plt.imshow(output_image.permute(1, 2, 0))
plt.show()

