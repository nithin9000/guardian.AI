import torch
from torchvision import transforms
from PIL import Image
from src.efficientNet import EfficientNetB7
import os

def load_model(checkpoint_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EfficientNetB7().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probability = output.item()
            prediction = "Real" if probability > 0.5 else "Fake"
            confidence = probability if probability > 0.5 else 1 - probability

        return prediction, confidence * 100
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

def process_directory(directory_path, model, device):
    import os
    results = []
    for image_name in os.listdir(directory_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            image_path = os.path.join(directory_path, image_name)
            prediction, confidence = predict_image(image_path, model, device)
            if prediction and confidence:
                results.append((image_name, prediction, confidence))
    return results

if __name__ == "__main__":
    try:
        # Load model
        model, device = load_model('checkpoints/model_exp1_best.pth')
        print(f"Model loaded successfully on {device}")

        # Test single image
        image_path = 'assets/person1.jpeg'  # Update with your image path
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
        else:
            prediction, confidence = predict_image(image_path, model, device)
            if prediction and confidence:
                print(f"\nSingle Image Analysis:")
                print(f"Image: {image_path}")
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2f}%")

        # Test directory of images (optional)
        test_dir = 'test_images'  # Update with your test directory
        if os.path.exists(test_dir):
            print(f"\nBatch Analysis:")
            results = process_directory(test_dir, model, device)
            for image_name, prediction, confidence in results:
                print(f"{image_name}: {prediction} (Confidence: {confidence:.2f}%)")

    except Exception as e:
        print(f"An error occurred: {str(e)}")