from cProfile import label
import torch
from torchvision import transforms
from PIL import Image
from .apps import PredAppConfig

TP = 0  # True Positives
TN = 0  # True Negatives
FP = 0  # False Positives
FN = 0  # False Negatives
total = 0

# Load the trained models
resnet50 = PredAppConfig.resnet50
swin_transformer = PredAppConfig.swin_transformer
effNetB6 = PredAppConfig.effNetB6  # Added EfficientNet-B6 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path)

    # Preprocess the image
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make predictions using the models
    with torch.no_grad():
        img = img.to(device)
        resnet50_op = resnet50(img)
        resnet50_pred = (resnet50_op > 0.5).float().item()

        swin_op = swin_transformer(img)
        swin_pred = (swin_op > 0.5).float().item()

        effNetB6_op = effNetB6(img)
        effNetB6_pred = (effNetB6_op > 0.5).float()

        #inception_op = inceptionv3(img)  # Make predictions with InceptionV3
        #inception_pred = (inception_op > 0.5).float().item()
ensemble_preds = torch.stack((resnet50_pred, swin_pred, effNetB6_pred), dim=1)
        # print(ensemble_preds)
        # Apply voting strategy (majority vote)
ensemble_prediction = torch.mode(ensemble_preds, dim=1)[0].squeeze(1)  # Get most frequent prediction (squeeze removes extra dimension)

        # print(ensemble_prediction)
        # ensemble_probabilities = (resnet50_probs + swin_probs) / 2.0
        # ensemble_predictions = (ensemble_probabilities > 0.6).float()

TP += ((ensemble_prediction == 1) & (label == 1)).sum().item()
TN += ((ensemble_prediction == 0) & (label == 0)).sum().item()
FP += ((ensemble_prediction == 1) & (label == 0)).sum().item()
FN += ((ensemble_prediction == 0) & (label == 1)).sum().item()

print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)
print('Total:', total)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / total
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

    #return resnet50_pred, swin_pred, effNetB6_pred
    

    
    
   # def load_inceptionv3_model():
    # Download model from Google Drive if not present
    #inception_model_path = os.path.join(settings.MODELS_PATH, 'inceptionv3_model.pth')
    #if not os.path.exists(inception_model_path):
      #  print("-"*30)
       # print("Downloading InceptionV3 Model from Google Drive.")
        #inception_file_id = '1iQKueWB_kYXn70HB7BVTi-HhRRX0PSQd'
        #download_file_from_google_drive(inception_file_id, inception_model_path)
    #else:
     #   pass
