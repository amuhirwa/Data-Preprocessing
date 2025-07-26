# User Identity and Product Recommendation System
## Formative 2 - Data Preprocessing Assignment

### Project Overview

This project implements a multimodal authentication and product recommendation system that combines facial recognition, voice verification, and customer behavior analysis to provide personalized product recommendations. The system follows a sequential authentication flow where users must pass both facial and voice verification before receiving product recommendations.

### System Architecture

The system implements the following flow:
1. **Face Recognition**: User submits facial image for identity verification
2. **Product Recommendation**: If recognized, system retrieves customer data for product prediction
3. **Voice Verification**: User provides voice sample to confirm transaction
4. **Access Control**: System grants access only if both face and voice match the same user

### Team Members and Contributions

**Task Division:**
- **Task 1 - Data Merge**: Ivan Shema
- **Task 2 - Image Data Collection and Processing**: Afsa Umutoniwase
- **Task 3 - Sound Data Collection and Processing**: Favour Ololade
- **Task 4 - Model Creation**: Alain Michael
- **Task 5 - System Demonstration**: Amandine Irakoze

**Detailed Contributions:**
- **Ivan Shema**: Data merging, customer social profiles and transactions integration, exploratory data analysis
- **Afsa Umutoniwase**: Image collection, facial expression processing, image augmentation, feature extraction
- **Favour Ololade**: Audio recording, sound processing, audio visualization, audio feature extraction
- **Alain Michael**: Model training, facial recognition model, voice verification model, product recommendation model
- **Amandine Irakoze**: System demonstration, CLI interface, unauthorized access simulation, testing scenarios

### Project Structure

```
Data-Preprocessing/
├── assets/
│   ├── audios/          # Audio samples for each team member
│   └── images/          # Facial images (neutral, smiling, surprised)
├── data/
│   ├── audio_features.csv           # Extracted audio features
│   ├── image_features.csv           # Extracted image features
│   ├── merged_customer_data.csv     # Merged customer dataset
│   ├── customer_social_profiles.csv # Social media profiles
│   └── customer_transactions.csv    # Transaction history
├── models/              # Trained model files
├── notebooks/           # Jupyter notebooks for analysis
├── scripts/             # Python implementation scripts
└── requirements.txt     # Dependencies
```

## 1. Data Merge and Feature Engineering

### 1.1 Dataset Overview

**Customer Social Profiles Dataset:**
- Contains customer engagement metrics across social media platforms
- Features: customer_id_new, social_media_platform, engagement_score, purchase_interest_score, review_sentiment

**Customer Transactions Dataset:**
- Contains historical purchase data
- Features: customer_id_legacy, transaction_id, purchase_amount, purchase_date, product_category, customer_rating

### 1.2 Merge Strategy

The datasets were merged using an inner join on `customer_id_new`, with the transaction dataset's legacy IDs converted to match the social profile format by adding "A" prefix. This approach was chosen because:

- The social profile dataset uses "customer_id_new" indicating it's the current format
- The transaction dataset uses "customer_id_legacy" indicating it's the old format
- Adding "A" prefix ensures compatibility between datasets

**Merge Results:**
- Original social profiles: 100 customers
- Original transactions: 150 transactions
- Merged dataset: 219 records (successful inner join)
- Features: 11 columns combining both datasets

### 1.3 Exploratory Data Analysis

**Social Media Platform Distribution:**
- Facebook: Most prevalent platform
- Twitter: Second most common
- LinkedIn, Instagram, TikTok: Lower representation

**Key Insights:**
- Strong correlation between engagement_score and purchase_interest_score
- Product categories show varied distribution (Electronics, Sports, Books, etc.)
- Customer ratings range from 1.1 to 5.0 with mean around 3.5

## 2. Image Data Collection and Processing

### 2.1 Image Collection

Each team member provided 3 facial expressions:
- **Neutral**: Baseline expression for recognition
- **Smiling**: Positive emotion state
- **Surprised**: High emotion state

**Image Specifications:**
- Format: JPG/PNG
- Naming convention: `{member}_{expression}.jpg`
- Quality: High resolution for feature extraction

### 2.2 Image Augmentation

Applied multiple augmentations per image to improve model robustness:

1. **Rotation**: 45-degree rotation to handle tilted faces
2. **Horizontal Flip**: Mirror image to increase dataset diversity
3. **Grayscale Conversion**: Monochrome version for texture analysis

### 2.3 Feature Extraction

**Histogram Features:**
- Color histograms for RGB channels
- Grayscale histograms for texture analysis
- 256-bin histograms per channel

**Deep Learning Embeddings:**
- ResNet-18 pretrained model
- Final layer embeddings (512 features)
- Normalized using ImageNet statistics

**Feature Storage:**
- Saved to `data/image_features.csv`
- Format: member, expression, embedding, histogram, augmentation_type

## 3. Audio Data Collection and Processing

### 3.1 Audio Collection

Each team member recorded 2 phrases:
- "Yes, approve" (approval phrase)
- "Confirm transaction" (confirmation phrase)

**Audio Specifications:**
- Format: M4A, MP3, MP4
- Duration: 2-5 seconds per phrase
- Quality: Clear speech, minimal background noise

### 3.2 Audio Visualization

**Waveform Analysis:**
- Time-domain representation
- Amplitude variations across time
- Speech pattern identification

**Spectrogram Analysis:**
- Frequency-domain representation
- Time-frequency energy distribution
- Phoneme identification

### 3.3 Audio Augmentation

Applied multiple augmentations to improve model generalization:

1. **Pitch Shift**: ±4 semitones to simulate different voices
2. **Time Stretch**: 1.2x speed variation
3. **Background Noise**: Low-level noise addition (0.005 factor)

### 3.4 Feature Extraction

**MFCC Features:**
- Mel-frequency cepstral coefficients
- 13 coefficients per frame
- Captures vocal tract characteristics

**Spectral Features:**
- Spectral roll-off: Frequency below which 85% of energy is contained
- Energy: Root mean square energy
- Zero crossing rate: Temporal characteristics

**Feature Storage:**
- Saved to `data/audio_features.csv`
- Format: member, phrase, mfcc_features, spectral_features, augmentation_type

## 4. Model Implementation

### 4.1 Facial Recognition Model

**Architecture:**
- Random Forest Classifier (100 estimators)
- Input: Combined embeddings + histogram features
- Preprocessing: StandardScaler + PCA (5 components)

**Performance Metrics:**
- Accuracy: 1.0
- F1-Score: 1.0
- Log Loss: 0.15

**Key Features:**
- Handles multiple expressions per person
- Robust to image augmentations
- Fast inference time

### 4.2 Voice Verification Model

**Architecture:**
- Random Forest Classifier (100 estimators)
- Input: MFCC + spectral features
- Preprocessing: StandardScaler

**Performance Metrics:**
- Accuracy: 1.0
- F1-Score: 1.0
- Log Loss: 0.40

**Key Features:**
- Speaker identification across phrases
- Robust to audio augmentations
- Handles different audio formats

### 4.3 Product Recommendation Model

**Architecture:**
- Random Forest Classifier (100 estimators)
- Input: Customer features (engagement, platform, sentiment, etc.)
- Output: Product category prediction

**Performance Metrics:**
- Accuracy: 0.523
- F1-Score: 0.519
- Log Loss: 1.207

**Key Features:**
- Multi-class classification (6 product categories)
- Considers customer behavior patterns
- Integrates social media engagement data

## 5. System Demonstration

### 5.1 Command Line Interface

The system provides an interactive CLI for testing:

```bash
# Run interactive demo
python scripts/system_demo.py

# Run automated simulation
python scripts/system_demo.py simulate
```

### 5.2 Simulation Scenarios

**Authorized Access:**
1. Valid face image → Recognition successful
2. Valid voice sample → Verification successful
3. Customer ID lookup → Product recommendation displayed

**Unauthorized Access Attempts:**
1. **Unknown Face**: Unauthorized image → Access denied
2. **Mismatched Voice**: Valid face + wrong voice → Access denied
3. **Unknown Customer**: Valid credentials + invalid ID → Error handling

### 5.3 Security Features

- **Threshold-based Recognition**: 80% confidence threshold for face recognition
- **Multi-factor Authentication**: Requires both face and voice match
- **Error Handling**: Graceful handling of invalid inputs

## 6. Evaluation and Results

### 6.1 Model Performance Summary

| Model | Accuracy | F1-Score | Log Loss |
|-------|----------|----------|----------|
| Facial Recognition | 1.0 | 1.0 | 0.15 |
| Voice Verification | 1.0 | 1.0 | 0.40 |
| Product Recommendation | 0.52 | 0.52 | 1.20 |

### 6.2 Multimodal Logic Evaluation

The system successfully implements sequential authentication:
- **Face Recognition**: Primary identity verification
- **Voice Verification**: Secondary confirmation
- **Product Recommendation**: Final output based on customer data

**Security Analysis:**
- False Positive Rate: <5% for face recognition
- False Negative Rate: <15% for voice verification
- Overall System Security: High (requires both factors)

### 6.3 Limitations and Future Improvements

**Current Limitations:**
- Limited dataset size (5 team members)
- Basic audio processing (no noise cancellation)
- Simple product recommendation logic

**Future Enhancements:**
- Larger training dataset
- Advanced audio preprocessing
- Deep learning models (CNN, RNN)
- Real-time processing capabilities

## 7. Technical Implementation

### 7.1 Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy, scikit-learn
- **Image Processing**: OpenCV, PIL, torchvision
- **Audio Processing**: librosa, soundfile, pydub
- **Machine Learning**: scikit-learn, joblib
- **Visualization**: matplotlib, seaborn

### 7.2 File Organization

**Scripts:**
- `audio_processing.py`: Audio feature extraction and augmentation
- `process_images.py`: Image processing and feature extraction
- `multi_modal_model_trainer.py`: Model training and evaluation
- `system_demo.py`: Interactive system demonstration
- `predictor.py`: Prediction interface

**Data Files:**
- `merged_customer_data.csv`: Combined customer dataset
- `image_features.csv`: Extracted image features
- `audio_features.csv`: Extracted audio features

**Models:**
- `face_recognition_model.pkl`: Trained facial recognition model
- `voice_verification_model.pkl`: Trained voice verification model
- `product_recommendation_model.pkl`: Trained product recommendation model

## 8. Conclusion

This project successfully demonstrates a complete multimodal authentication and recommendation system. Key achievements include:

1. **Successful Data Integration**: Merged customer social and transaction data
2. **Robust Feature Engineering**: Extracted meaningful features from images and audio
3. **Effective Model Training**: Achieved good performance across all three models
4. **Working System Demo**: Functional CLI interface with security features
5. **Comprehensive Documentation**: Detailed implementation and evaluation

The system provides a solid foundation for real-world applications requiring secure, multimodal user authentication and personalized recommendations.

## 9. Usage Instructions

### 9.1 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run image processing
python scripts/process_images.py

# Run audio processing
python scripts/audio_processing.py

# Train models
python scripts/multi_modal_model_trainer.py

# Run system demo
python scripts/system_demo.py
```

### 9.2 Testing

```bash
# Interactive mode
python scripts/system_demo.py

# Automated simulation
python scripts/system_demo.py simulate
```

### 9.3 File Paths

When testing, use the following file paths:
- Images: `assets/images/{member}_{expression}.jpg`
- Audio: `assets/audios/{member}_{phrase}.{format}`
- Customer ID: Use IDs from `merged_customer_data.csv` (e.g., "A181")

---

**Repository**: https://github.com/amuhirwa/data-preprocessing
**Video Demo**: https://youtu.be/k_OLqRms91M
**Report**: This README serves as the comprehensive project report
