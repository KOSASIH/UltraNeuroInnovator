import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define the neural interface
class NeuralInterface(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralInterface, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Load the pre-trained CNN model
cnn_model = models.vgg16(pretrained=True).features.to(device).eval()

# Define the style transfer function
def style_transfer(content_image, style_image, num_steps=300, style_weight=1000000, content_weight=1):
    # Load and preprocess the content and style images
    content = load_image(content_image).to(device)
    style = load_image(style_image, shape=content.shape[-2:]).to(device)
    
    # Initialize the generated image as a copy of the content image
    generated = content.clone().requires_grad_(True).to(device)
    
    # Extract features from the pre-trained CNN model
    content_features = get_features(content, cnn_model)
    style_features = get_features(style, cnn_model)
    
    # Compute the gram matrices of the style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Set up the optimizer
    optimizer = optim.Adam([generated], lr=0.01)
    
    for step in range(num_steps):
        # Extract features from the generated image
        generated_features = get_features(generated, cnn_model)
        
        # Compute the content loss
        content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # Compute the style loss
        style_loss = 0
        for layer in style_weights:
            generated_gram = gram_matrix(generated_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((generated_gram - style_gram) ** 2)
            style_loss += style_weights[layer] * layer_style_loss
        
        # Compute the total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Update the generated image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Clamp the pixel values to the range [0, 1]
        generated.data.clamp_(0, 1)
        
        # Print the loss at every 50 steps
        if (step+1) % 50 == 0:
            print(f"Step [{step+1}/{num_steps}], Total Loss: {total_loss.item()}")
    
    # Return the stylized image
    return generated.detach().cpu()

# Helper functions
def load_image(image_path, shape=None):
    image = Image.open(image_path).convert('RGB')
    if shape is not None:
        image = image.resize(shape)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image.to(device)

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers:
            features['content'] = x
        if name in style_layers:
            features['style'] = x
    return features

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Define the content and style layers
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the style weights for each style layer
style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}

# Perform style transfer on an image
content_image = 'content.jpg'
style_image = 'style.jpg'
stylized_image = style_transfer(content_image, style_image, num_steps=300, style_weight=1000000, content_weight=1)

# Save the stylized image
output_image = 'stylized.jpg'
transforms.ToPILImage()(stylized_image.squeeze(0)).save(output_image)
