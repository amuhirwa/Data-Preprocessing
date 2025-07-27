#!/usr/bin/env python3
"""
Real Multimodal Authentication System Demo
Works with actual image_features.csv, audio_features.csv, and merged datasets
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import cv2
import librosa
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RealMultimodalAuthSystem:
    def __init__(self, data_dir="./data", models_dir="./models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Models
        self.face_model = None
        self.voice_model = None
        self.product_model = None
        
        # Data
        self.image_features_df = None
        self.audio_features_df = None
        self.merged_dataset = None
        
        # Scalers
        self.face_scaler = StandardScaler()
        self.voice_scaler = StandardScaler()
        self.product_scaler = StandardScaler()
        
        # Label encoders
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        
    def load_datasets(self):
        """Load the actual CSV datasets"""
        print("📂 Loading datasets...")
        
        try:
            # Load image features
            image_path = self.data_dir / "image_features.csv"
            if image_path.exists():
                self.image_features_df = pd.read_csv(image_path)
                print(f"✅ Loaded image features: {self.image_features_df.shape}")
            else:
                print(f"❌ Image features file not found: {image_path}")
                return False
                
            # Load audio features  
            audio_path = self.data_dir / "audio_features.csv"
            if audio_path.exists():
                self.audio_features_df = pd.read_csv(audio_path)
                print(f"✅ Loaded audio features: {self.audio_features_df.shape}")
            else:
                print(f"❌ Audio features file not found: {audio_path}")
                return False
                
            # Load merged dataset
            merged_path = self.data_dir / "merged_customer_data.csv"
            if merged_path.exists():
                self.merged_dataset = pd.read_csv(merged_path)
                print(f"✅ Loaded merged dataset: {self.merged_dataset.shape}")
            else:
                print(f"❌ Merged dataset file not found: {merged_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def prepare_face_recognition_data(self):
        """Prepare data for face recognition model"""
        print("🔄 Preparing face recognition data...")
        
        # Assume image_features.csv has columns: user_id, feature_1, feature_2, ..., feature_n
        if 'user_id' not in self.image_features_df.columns:
            print("❌ 'user_id' column not found in image features")
            return None, None
            
        # Separate features and labels
        feature_cols = [col for col in self.image_features_df.columns if col != 'user_id']
        X = self.image_features_df[feature_cols].values
        y = self.image_features_df['user_id'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.face_scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.user_encoder.fit_transform(y)
        
        print(f"✅ Face data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_encoded
    
    def prepare_voice_verification_data(self):
        """Prepare data for voice verification model"""
        print("🔄 Preparing voice verification data...")
        
        # Assume audio_features.csv has columns: user_id, approved, mfcc_1, mfcc_2, ..., spectral_rolloff, energy
        required_cols = ['user_id', 'approved']
        for col in required_cols:
            if col not in self.audio_features_df.columns:
                print(f"❌ '{col}' column not found in audio features")
                return None, None
        
        # Separate features and labels
        feature_cols = [col for col in self.audio_features_df.columns 
                       if col not in ['user_id', 'approved']]
        X = self.audio_features_df[feature_cols].values
        y = self.audio_features_df['approved'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.voice_scaler.fit_transform(X)
        
        print(f"✅ Voice data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def prepare_product_recommendation_data(self):
        """Prepare data for product recommendation model"""
        print("🔄 Preparing product recommendation data...")
        
        # Assume merged dataset has product column and various features
        if 'product' not in self.merged_dataset.columns:
            print("❌ 'product' column not found in merged dataset")
            return None, None
            
        # Separate features and target
        feature_cols = [col for col in self.merged_dataset.columns 
                       if col not in ['product', 'user_id', 'customer_id']]
        X = self.merged_dataset[feature_cols].values
        y = self.merged_dataset['product'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.product_scaler.fit_transform(X)
        
        # Encode target
        y_encoded = self.product_encoder.fit_transform(y)
        
        print(f"✅ Product data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_encoded
    
    def train_models(self):
        """Train all three models"""
        print("\n🤖 Training models...")
        
        # Train face recognition model
        X_face, y_face = self.prepare_face_recognition_data()
        if X_face is not None:
            self.face_model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_face, y_face, test_size=0.2, random_state=42)
            
            self.face_model.fit(X_train, y_train)
            y_pred = self.face_model.predict(X_test)
            
            face_accuracy = accuracy_score(y_test, y_pred)
            face_f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"✅ Face Recognition Model - Accuracy: {face_accuracy:.3f}, F1-Score: {face_f1:.3f}")
            
            # Save model
            joblib.dump(self.face_model, self.models_dir / "face_model.pkl")
            joblib.dump(self.face_scaler, self.models_dir / "face_scaler.pkl")
            joblib.dump(self.user_encoder, self.models_dir / "user_encoder.pkl")
        
        # Train voice verification model
        X_voice, y_voice = self.prepare_voice_verification_data()
        if X_voice is not None:
            self.voice_model = LogisticRegression(random_state=42, max_iter=1000)
            X_train, X_test, y_train, y_test = train_test_split(X_voice, y_voice, test_size=0.2, random_state=42)
            
            self.voice_model.fit(X_train, y_train)
            y_pred = self.voice_model.predict(X_test)
            
            voice_accuracy = accuracy_score(y_test, y_pred)
            voice_f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"✅ Voice Verification Model - Accuracy: {voice_accuracy:.3f}, F1-Score: {voice_f1:.3f}")
            
            # Save model
            joblib.dump(self.voice_model, self.models_dir / "voice_model.pkl")
            joblib.dump(self.voice_scaler, self.models_dir / "voice_scaler.pkl")
        
        # Train product recommendation model
        X_product, y_product = self.prepare_product_recommendation_data()
        if X_product is not None:
            self.product_model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_product, y_product, test_size=0.2, random_state=42)
            
            self.product_model.fit(X_train, y_train)
            y_pred = self.product_model.predict(X_test)
            
            product_accuracy = accuracy_score(y_test, y_pred)
            product_f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"✅ Product Recommendation Model - Accuracy: {product_accuracy:.3f}, F1-Score: {product_f1:.3f}")
            
            # Save model
            joblib.dump(self.product_model, self.models_dir / "product_model.pkl")
            joblib.dump(self.product_scaler, self.models_dir / "product_scaler.pkl")
            joblib.dump(self.product_encoder, self.models_dir / "product_encoder.pkl")
    
    def load_trained_models(self):
        """Load pre-trained models if they exist"""
        print("📥 Loading trained models...")
        
        model_files = {
            'face_model.pkl': 'face_model',
            'voice_model.pkl': 'voice_model', 
            'product_model.pkl': 'product_model'
        }
        
        scaler_files = {
            'face_scaler.pkl': 'face_scaler',
            'voice_scaler.pkl': 'voice_scaler',
            'product_scaler.pkl': 'product_scaler'
        }
        
        encoder_files = {
            'user_encoder.pkl': 'user_encoder',
            'product_encoder.pkl': 'product_encoder'
        }
        
        try:
            # Load models
            for file, attr in model_files.items():
                path = self.models_dir / file
                if path.exists():
                    setattr(self, attr, joblib.load(path))
                    print(f"✅ Loaded {attr}")
                else:
                    print(f"⚠️  {file} not found, will need to train")
                    return False
            
            # Load scalers
            for file, attr in scaler_files.items():
                path = self.models_dir / file
                if path.exists():
                    setattr(self, attr, joblib.load(path))
            
            # Load encoders
            for file, attr in encoder_files.items():
                path = self.models_dir / file
                if path.exists():
                    setattr(self, attr, joblib.load(path))
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def extract_image_features_from_file(self, image_path):
        """Extract features from a new image file"""
        try:
            print(f"📷 Processing image: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"❌ Image file not found: {image_path}")
                return None
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print("❌ Could not load image")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract basic features (you should replace this with your actual feature extraction)
            # For demo, using histogram and basic statistics
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features = []
            
            # Histogram features
            features.extend(hist.flatten()[:50])  # First 50 histogram bins
            
            # Statistical features
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray)
            ])
            
            # Pad or truncate to match training data size
            expected_size = self.image_features_df.shape[1] - 1  # -1 for user_id column
            if len(features) < expected_size:
                features.extend([0] * (expected_size - len(features)))
            else:
                features = features[:expected_size]
            
            print("✅ Image features extracted")
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error extracting image features: {e}")
            return None
    
    def extract_audio_features_from_file(self, audio_path):
        """Extract features from a new audio file"""
        try:
            print(f"🎤 Processing audio: {audio_path}")
            
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                return None
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Extract additional features
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            energy = np.mean(librosa.feature.rms(y=y))
            
            # Combine features
            features = list(mfcc_mean) + [spectral_rolloff, zero_crossing_rate, energy]
            
            # Pad or truncate to match training data size
            expected_size = len([col for col in self.audio_features_df.columns 
                               if col not in ['user_id', 'approved']])
            if len(features) < expected_size:
                features.extend([0] * (expected_size - len(features)))
            else:
                features = features[:expected_size]
            
            print("✅ Audio features extracted")
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error extracting audio features: {e}")
            return None
    
    def authenticate_face(self, image_path):
        """Perform face recognition on a new image"""
        features = self.extract_image_features_from_file(image_path)
        if features is None:
            return False, None, 0.0
        
        # Scale features
        features_scaled = self.face_scaler.transform(features)
        
        # Predict
        prediction = self.face_model.predict(features_scaled)[0]
        probabilities = self.face_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Decode user
        user_name = self.user_encoder.inverse_transform([prediction])[0]
        
        # Set threshold for authentication
        is_authorized = confidence > 0.6
        
        return is_authorized, user_name, confidence
    
    def verify_voice(self, audio_path):
        """Perform voice verification on a new audio file"""
        features = self.extract_audio_features_from_file(audio_path)
        if features is None:
            return False, 0.0
        
        # Scale features
        features_scaled = self.voice_scaler.transform(features)
        
        # Predict
        prediction = self.voice_model.predict(features_scaled)[0]
        probabilities = self.voice_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        is_approved = prediction == 1 or prediction == 'approved'  # Handle different encodings
        
        return is_approved, confidence
    
    def get_product_recommendation(self, user_id):
        """Get product recommendation for authenticated user"""
        # Find user data in merged dataset
        user_data = self.merged_dataset[self.merged_dataset.get('user_id', self.merged_dataset.columns[0]) == user_id]
        
        if user_data.empty and len(self.merged_dataset) > 0:
            # Use first row as fallback
            user_data = self.merged_dataset.iloc[:1]
        
        if user_data.empty:
            return "No recommendation available", 0.0
        
        # Extract features
        feature_cols = [col for col in self.merged_dataset.columns 
                       if col not in ['product', 'user_id', 'customer_id']]
        features = user_data[feature_cols].values
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Scale features
        features_scaled = self.product_scaler.transform(features)
        
        # Predict
        prediction = self.product_model.predict(features_scaled)[0]
        probabilities = self.product_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Decode product
        try:
            product_name = self.product_encoder.inverse_transform([prediction])[0]
        except:
            product_name = f"Product_{prediction}"
        
        return product_name, confidence
    
    def simulate_unauthorized_attempt(self, image_path, audio_path):
        """Simulate unauthorized access attempt with real files"""
        print("\n" + "="*60)
        print("🚫 SIMULATING UNAUTHORIZED ATTEMPT")
        print("="*60)
        
        # Try face authentication
        print("\n🔍 Attempting face authentication...")
        is_authorized, user_name, confidence = self.authenticate_face(image_path)
        
        print(f"Face recognition confidence: {confidence:.2%}")
        
        if not is_authorized:
            print("❌ FACE AUTHENTICATION FAILED")
            print("🚫 ACCESS DENIED - Unauthorized user")
            return False
        else:
            print(f"✅ Face recognized as: {user_name}")
            
            # Even if face is recognized, try voice verification
            print("\n🎤 Attempting voice verification...")
            is_voice_approved, voice_confidence = self.verify_voice(audio_path)
            
            print(f"Voice verification confidence: {voice_confidence:.2%}")
            
            if not is_voice_approved:
                print("❌ VOICE VERIFICATION FAILED")
                print("🚫 ACCESS DENIED - Voice not approved")
                return False
            else:
                print("⚠️  Both face and voice passed, but this was meant to be unauthorized!")
                return True
    
    def simulate_full_transaction(self, image_path, audio_path):
        """Simulate complete transaction with real files"""
        print("\n" + "="*60)
        print("🔐 STARTING FULL TRANSACTION SIMULATION")
        print("="*60)
        
        # Step 1: Face Authentication
        print("\n🔍 Step 1: Face Authentication")
        print("-" * 40)
        
        is_authorized, user_name, face_confidence = self.authenticate_face(image_path)
        
        if not is_authorized:
            print(f"❌ FACE AUTHENTICATION FAILED (Confidence: {face_confidence:.2%})")
            print("🚫 ACCESS DENIED")
            return False
        
        print(f"✅ Face authenticated successfully!")
        print(f"👤 Welcome, {user_name}!")
        print(f"📊 Confidence: {face_confidence:.2%}")
        
        # Step 2: Voice Verification
        print("\n🎤 Step 2: Voice Verification")
        print("-" * 40)
        
        is_voice_approved, voice_confidence = self.verify_voice(audio_path)
        
        if not is_voice_approved:
            print(f"❌ VOICE VERIFICATION FAILED (Confidence: {voice_confidence:.2%})")
            print("🚫 TRANSACTION DENIED")
            return False
        
        print("✅ Voice verified successfully!")
        print(f"📊 Confidence: {voice_confidence:.2%}")
        
        # Step 3: Product Recommendation
        print("\n🛍️  Step 3: Product Recommendation")
        print("-" * 40)
        
        product, product_confidence = self.get_product_recommendation(user_name)
        
        print(f"🎯 Recommended Product: {product}")
        print(f"📊 Confidence Score: {product_confidence:.2%}")
        
        # Step 4: Transaction Completion
        print("\n✅ TRANSACTION COMPLETED SUCCESSFULLY!")
        print(f"🎉 {user_name}, your recommended product '{product}' is ready!")
        
        return True
    
    def show_system_stats(self):
        """Display real system statistics"""
        print("\n📊 SYSTEM STATISTICS")
        print("-" * 40)
        
        if self.image_features_df is not None:
            unique_users = self.image_features_df['user_id'].nunique() if 'user_id' in self.image_features_df.columns else 0
            print(f"👥 Authorized Users: {unique_users}")
            print(f"📷 Image Samples: {len(self.image_features_df)}")
        
        if self.audio_features_df is not None:
            print(f"🎤 Audio Samples: {len(self.audio_features_df)}")
        
        if self.merged_dataset is not None:
            unique_products = self.merged_dataset['product'].nunique() if 'product' in self.merged_dataset.columns else 0
            print(f"🛍️  Product Catalog: {unique_products} items")
            print(f"📦 Total Records: {len(self.merged_dataset)}")
        
        print(f"\n🤖 MODEL INFORMATION:")
        print(f"🎯 Face Model: {'Loaded' if self.face_model else 'Not loaded'}")
        print(f"🎤 Voice Model: {'Loaded' if self.voice_model else 'Not loaded'}")
        print(f"📦 Product Model: {'Loaded' if self.product_model else 'Not loaded'}")
    
    def run_interactive_demo(self):
        """Run interactive command-line demo"""
        print("🚀 REAL MULTIMODAL AUTHENTICATION SYSTEM")
        print("=" * 50)
        
        # Load datasets
        if not self.load_datasets():
            print("❌ Failed to load datasets. Please check your data directory.")
            return
        
        # Try to load existing models, otherwise train new ones
        if not self.load_trained_models():
            print("🔄 Training new models...")
            self.train_models()
        
        while True:
            print("\n" + "-" * 50)
            print("DEMO OPTIONS:")
            print("1. Simulate Full Transaction (with real files)")
            print("2. Simulate Unauthorized Attempt (with real files)")
            print("3. View System Statistics")
            print("4. Train New Models")
            print("5. Exit")
            print("-" * 50)
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                image_path = input("Enter path to face image: ").strip()
                audio_path = input("Enter path to voice audio: ").strip()
                self.simulate_full_transaction(image_path, audio_path)
                
            elif choice == "2":
                image_path = input("Enter path to unauthorized face image: ").strip()
                audio_path = input("Enter path to unauthorized voice audio: ").strip()
                self.simulate_unauthorized_attempt(image_path, audio_path)
                
            elif choice == "3":
                self.show_system_stats()
                
            elif choice == "4":
                print("🔄 Retraining models...")
                self.train_models()
                
            elif choice == "5":
                print("👋 Thank you for using the Real Multimodal Auth System!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    """Main function to run the demonstration"""
    print("Initializing Real Multimodal Authentication System...")
    
    # You can customize these paths
    data_directory = "./data"  # Directory containing your CSV files
    models_directory = "./models"  # Directory to save/load models
    
    # Create system instance
    auth_system = RealMultimodalAuthSystem(data_directory, models_directory)
    
    # Run interactive demo
    auth_system.run_interactive_demo()

if __name__ == "__main__":
    main()