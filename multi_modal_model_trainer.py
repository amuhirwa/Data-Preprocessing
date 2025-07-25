import os
from audio_processing import display_audio_samples, process_audio_augmentations
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import joblib
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import ast
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.image_scaler = StandardScaler()
        self.image_pca = PCA(n_components=5)
        self.audio_scaler = StandardScaler()

    def load_data(self):
        """
        Load all processed datasets
        """
        image_features = pd.read_csv('image_features.csv')
        audio_features = pd.read_csv('audio_features.csv')

        # Optional: product recommendation data
        X_train = None
        X_test = None
        y_train = None
        y_test = None

        return image_features, audio_features, X_train, X_test, y_train, y_test

    def train_facial_recognition_model(self, image_features):
        print("Training Facial Recognition Model...")

        # Parse features
        image_features['embedding'] = image_features['embedding'].apply(ast.literal_eval)
        image_features['histogram'] = image_features['histogram'].apply(ast.literal_eval)

        # Pad histogram to fixed length
        def pad_histogram(hist, target_len=768):
            return hist + [0] * (target_len - len(hist))

        image_features['histogram_padded'] = image_features['histogram'].apply(lambda h: pad_histogram(h, 768))
        image_features['features'] = image_features.apply(
            lambda row: row['embedding'] + row['histogram_padded'], axis=1
        )

        X = np.array(image_features['features'].tolist())
        y = image_features['member']

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['face'] = le

        # Scale and reduce
        X_scaled = self.image_scaler.fit_transform(X)
        X_reduced = self.image_pca.fit_transform(X_scaled)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        y_pred_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)

        print(f"Facial Recognition - Accuracy: {acc:.3f}, F1-Score: {f1:.3f}, Log Loss: {loss:.3f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        self.models['face_recognition'] = model
        joblib.dump(model, 'models/face_recognition_model.pkl')
        joblib.dump(le, 'models/face_label_encoder.pkl')
        joblib.dump(self.image_scaler, 'models/image_scaler.pkl')
        joblib.dump(self.image_pca, 'models/image_pca.pkl')

        return model, acc, f1, loss

    def train_voice_verification_model(self, audio_features):
        print("Training Voice Verification Model...")

        # Extract numerical features
        exclude = ['member', 'phrase', 'augmentation']
        feature_cols = [col for col in audio_features.columns if col not in exclude]
        X = audio_features[feature_cols].values
        y = audio_features['member']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['voice'] = le

        # Scale
        X_scaled = self.audio_scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        y_pred_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)

        print(f"Voice Verification - Accuracy: {acc:.3f}, F1-Score: {f1:.3f}, Log Loss: {loss:.3f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Save
        self.models['voice_verification'] = model
        joblib.dump(model, 'models/voice_verification_model.pkl')
        joblib.dump(le, 'models/voice_label_encoder.pkl')
        joblib.dump(self.audio_scaler, 'models/audio_scaler.pkl')

        return model, acc, f1, loss

    def train_product_recommendation_model(self, product_df):
        print("Training Product Recommendation Model...")

        # Drop identifiers
        product_df = product_df.drop(columns=['customer_id_new', 'transaction_id'])

        # Handle date
        product_df['purchase_date'] = pd.to_datetime(product_df['purchase_date'])
        product_df['purchase_month'] = product_df['purchase_date'].dt.month
        product_df = product_df.drop(columns=['purchase_date'])  # original date not needed

        # Encode categorical features
        cat_cols = ['social_media_platform', 'review_sentiment']
        product_df = pd.get_dummies(product_df, columns=cat_cols, drop_first=True)

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(product_df['product_category'])
        self.label_encoders['product'] = le
        X = product_df.drop(columns=['product_category'])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)

        # Compute log loss
        loss = log_loss(y_test, y_pred_proba)


        print(f"Product Recommendation - Accuracy: {acc:.3f}, F1-Score: {f1:.3f}, Log Loss: {loss:.3f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Save model
        self.models['product_recommendation'] = model
        joblib.dump(model, 'models/product_recommendation_model.pkl')
        joblib.dump(le, 'models/product_label_encoder.pkl')

        return model, acc, f1, loss


    def predict_from_image(image_path):
      # Load model + preprocessing
      model = joblib.load('models/face_recognition_model.pkl')
      scaler = joblib.load('models/image_scaler.pkl')
      pca = joblib.load('models/image_pca.pkl')
      le = joblib.load('models/face_label_encoder.pkl')

      # Set device
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      # Load pretrained model
      base_model = models.resnet18(weights='DEFAULT')
      embedding_model = torch.nn.Sequential(*list(base_model.children())[:-1])
      embedding_model.to(device)
      embedding_model.eval()

      # Transforms
      transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

      # Load and process image
      img = cv2.imread(image_path)
      if img is None:
          raise ValueError("Could not read image!")

      if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

      # Extract histogram
      def extract_histogram(img):
          if len(img.shape) == 2:
              hist = cv2.calcHist([img], [0], None, [256], [0, 256])
              return hist.flatten()
          else:
              hist = []
              for i in range(3):
                  h = cv2.calcHist([img], [i], None, [256], [0, 256])
                  hist.extend(h.flatten())
              return np.array(hist)

      hist = extract_histogram(img)
      hist = np.pad(hist, (0, max(0, 768 - len(hist))), constant_values=0)

      # Extract embedding
      pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      input_tensor = transform(pil_img).unsqueeze(0).to(device)

      with torch.no_grad():
          emb = embedding_model(input_tensor).cpu().numpy().flatten()

      combined = np.concatenate([emb, hist])
      X_scaled = scaler.transform([combined])
      X_reduced = pca.transform(X_scaled)
      y_pred = model.predict(X_reduced)
      probabilities = model.predict_proba(X_reduced)[0]

      predicted_label = le.inverse_transform(y_pred)[0]
      return predicted_label, probabilities
    
    def predict_from_audio(audio_path):
      y, sr = display_audio_samples(audio_path, f'None - none', show_plots=False)

      # Load models
      model = joblib.load('models/voice_verification_model.pkl')
      scaler = joblib.load('models/audio_scaler.pkl')
      le = joblib.load('models/voice_label_encoder.pkl')

      # Extract features like in training
      phrase_features = process_audio_augmentations(y, sr, "michael", "approve")
      exclude = ['member', 'phrase', 'augmentation']
      filtered_features = {k: v for k, v in phrase_features[0].items() if k not in exclude}
      X = np.array(list(filtered_features.values())).reshape(1, -1)
      X_scaled = scaler.transform(X)

      y_pred = model.predict(X_scaled)
      predicted_label = le.inverse_transform(y_pred)[0]
      return predicted_label

    def evaluate_models(self):
        print("\n======== MODEL EVALUATION SUMMARY ========")
        for name, model in self.models.items():
            print(f"\n{name.upper()} Model - {type(model).__name__}")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                print(f"Top 3 Feature Importances: {sorted(importances, reverse=True)[:3]}")

def main():
    os.makedirs('models', exist_ok=True)
    trainer = ModelTrainer()

    print("Loading data...")
    image_features, audio_features, X_train, X_test, y_train, y_test = trainer.load_data()

    face_model, face_acc, face_f1, face_loss = trainer.train_facial_recognition_model(image_features)
    voice_model, voice_acc, voice_f1, voice_loss = trainer.train_voice_verification_model(audio_features)
    product_model, product_acc, product_f1, product_loss = trainer.train_product_recommendation_model(X_train, X_test, y_train, y_test)

    trainer.evaluate_models()

    summary = {
        'facial_recognition': {'accuracy': face_acc, 'f1_score': face_f1, 'loss': face_loss},
        'voice_verification': {'accuracy': voice_acc, 'f1_score': voice_f1, 'loss': voice_loss},
        'product_recommendation': {'accuracy': product_acc, 'f1_score': product_f1, 'loss': product_loss}
    }

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv('models/training_summary.csv')
    print("\n======== TRAINING COMPLETE ========")

if __name__ == "__main__":
    main()