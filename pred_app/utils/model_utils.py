import torchvision.models as models
import torch.nn as nn
import torch
import os
import timm
import gdown
from django.conf import settings


# Define functions to load your trained models (replace with your specific code)
def load_resnet50_model():
    # Download model from gdrive if not present
    resnet_model_path = os.path.join(settings.MODELS_PATH, 'adversarial_model1.pth')
    if not os.path.exists(resnet_model_path):
        print("-"*30)
        print("Downloading Swin Model from gdrive.")
        resnet_file_id = '1HGZc0sM_dEksFeg7Ed9rORUTBBitSisQ'
        download_file_from_google_drive(resnet_file_id, resnet_model_path)
    else:
        pass
    # Load the pre-trained ResNet50 model without pre-trained weights
    model = models.resnet50(pretrained=False)  # Load without pre-trained weights
    # Freeze model parameters (if necessary, only if you also froze them during training)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer (to match saved model architecture)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()  # Apply Sigmoid activation function
    )

    # resnet model name model.pth
    model_state = torch.load(resnet_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)


    # Load the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def load_swin_transformer_model():
    # Download model from gdrive if not present
    swin_model_path = os.path.join(settings.MODELS_PATH, 'adversarial_swin_transformer_model.pth')
    if not os.path.exists(swin_model_path):
        print("-"*30)
        print("Downloading Swin Model from gdrive.")
        swin_file_id = '1zlfLbU3HDBr_gkAxd9eSLkxHPvZ7CZLy'
        download_file_from_google_drive(swin_file_id, swin_model_path)
    else:
        pass
    # Load the Swin Transformer model without pre-trained weights
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=1)

    # Freeze model parameters (if necessary, only if you also froze them during training)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer (to match saved model configuration)
    for param in model.head.parameters():
        param.requires_grad = True

    # Load the saved model state
    model_state = torch.load(swin_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

    return model
def load_EffNetB6_model():
    # Download model from gdrive if not present
    effnetb6_model_path = os.path.join(settings.MODELS_PATH, 'EffBetB6_0504.pth')
    if not os.path.exists(effnetb6_model_path):
        print("-"*30)
        print("Downloading EfficientNet-B6 Model from gdrive.")
        effnetb6_file_id = '1abcdefghi_jklmnopqrstuvwxyz'
        download_file_from_google_drive(effnetb6_file_id, effnetb6_model_path)
    else:
        pass

    # Load pre-trained EfficientNet-B6 model
    EffNetB6 = models.efficientnet_b6(pretrained=True)

    # Freeze the pre-trained weights
    for param in EffNetB6.parameters():
        param.requires_grad = False

    # Modify the classifier part of the model
    num_features_in = EffNetB6.classifier[1].in_features  # Get number of input features to the last layer
    EffNetB6.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(num_features_in, 1),  # Replace last layer with a linear layer for 1 output (binary classification)
        torch.nn.Sigmoid()  # Add a Sigmoid activation function
    )

    # Load the saved model state
    model_state = torch.load(effnetb6_model_path, map_location=torch.device('cpu'))
    EffNetB6.load_state_dict(model_state)

    return EffNetB6
    
def download_file_from_google_drive(file_id, output_path):
    """
    Download a file from Google Drive.
    
    Args:
    - file_id (str): The file ID of the file to download from Google Drive.
    - output_path (str): The path where the downloaded file will be saved.
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
