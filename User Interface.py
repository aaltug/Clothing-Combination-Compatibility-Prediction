import sys
sys.path.append('C:\\Users\\altu_\\OneDrive\\Masaüstü\\kombin\\venv\\Lib\\site-packages')

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy path: {np.__file__}")
except ImportError as e:
    print(f"NumPy import error: {str(e)}")
    raise

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from joblib import load, dump
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Klasör yoksa oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# VGG19 modelini yükle
model_vgg = vgg19(pretrained=True)
model_vgg.classifier = torch.nn.Sequential(*list(model_vgg.classifier.children())[:-3])
model_vgg.eval()

# XGBoost modelini yükle
try:
    print("NumPy kullanılabilir durumda, model yükleniyor...")
    model = load("model.joblib")
    
    print("Model başarıyla yüklendi")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {str(e)}")
    print(f"NumPy durumu: {np.__version__}")
    raise

# Görsel ön işleme
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model_vgg(input_tensor)
    return features.squeeze().numpy()

def predict_compatibility(upper_features, lower_features, third_features):
    try:
        # Özelliklerin ortalamasını al (yan yana eklemek yerine)
        combined_features = np.mean([upper_features, lower_features, third_features], axis=0)
        print(f"Özellik boyutu: {combined_features.shape}")
        # Skor hesapla
        score = model.predict_proba([combined_features])[0][1]
        return float(score)
    except Exception as e:
        print(f"Tahmin yapılırken hata oluştu: {str(e)}")
        print(f"Özellik boyutları - upper: {upper_features.shape}, lower: {lower_features.shape}, third: {third_features.shape}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'upper' not in request.files or 'lower' not in request.files or 'third' not in request.files:
        return jsonify({'error': 'Lütfen tüm parçaları yükleyin'}), 400

    upper_file = request.files['upper']
    lower_file = request.files['lower']
    third_file = request.files['third']

    try:
        # Görüntüleri yükle ve özellik çıkar
        upper_img = Image.open(upper_file).convert('RGB')
        lower_img = Image.open(lower_file).convert('RGB')
        third_img = Image.open(third_file).convert('RGB')

        upper_features = extract_features(upper_img)
        lower_features = extract_features(lower_img)
        third_features = extract_features(third_img)

        # Uyumluluk skorunu hesapla
        compatibility_score = predict_compatibility(upper_features, lower_features, third_features)

        # Görüntüleri base64'e çevir
        def image_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'compatibility_score': compatibility_score,
            'upper_image': image_to_base64(upper_img),
            'lower_image': image_to_base64(lower_img),
            'third_image': image_to_base64(third_img)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 