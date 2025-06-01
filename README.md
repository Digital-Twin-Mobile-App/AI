
# DIGITAL TWIN MOBILE APP AI

## Overview

The AI component is a critical part of the Digital Twin Mobile App, providing automated analysis of plant images to detect species, growth stages, and physical characteristics. It operates as a separate service that communicates with the main backend through a REST API.

## Features

- **Plant Species Identification**: Classifies plants into specific species using a CNN model
- **Growth Stage Detection**: Analyzes plant development stage (Germination, Vegetation, Flowering)
- **Height Ratio Calculation**: Measures the relative height of plants for growth tracking
- **Confidence Scoring**: Provides confidence levels for all predictions

## Technical Implementation

### AI Model Architecture

The AI system uses a TensorFlow-based model with the following components:
- CNN for species classification (pre-trained model loaded from `oppd_model.h5`)
- Computer vision techniques for growth stage analysis using OpenCV
- Height ratio calculation through contour detection

### Image Processing Pipeline

1. **Preprocessing**:
   - Resize images to 224x224 pixels
   - Normalize pixel values to [0,1]
   - Convert to RGB color space

2. **Species Classification**:
   - Pass processed image through CNN model
   - Extract species prediction and confidence score

3. **Growth Stage Analysis**:
   - Convert image to HSV color space
   - Create mask to isolate plant regions
   - Apply Gaussian blur to reduce noise
   - Find contours of plant regions
   - Calculate height ratio relative to image size
   - Determine growth stage based on height ratio:
     - < 10%: Early sprout (Germination)
     - 10-25%: Developing sprout (Vegetation)
     - > 25%: Mature sprout (Flowering)

## Integration with Backend

### Asynchronous Processing

The backend uses RabbitMQ to handle AI processing asynchronously:

1. When a user uploads a plant image, it's stored temporarily and a message is sent to the AI prediction queue
2. The `AIPredictionConsumer` processes the queue messages by:
   - Sending the image to the AI service via REST API
   - Processing the prediction results
   - Updating the plant and image records with AI analysis data
   - Creating notifications if plant stage changes are detected

### Data Flow

```
User Upload → Image Storage → RabbitMQ Queue → AI Service → 
Prediction Results → Database Update → User Notification
```

### Configuration

The AI service URL is configured in the application properties:
```
ai.service.url=${AI_SERVICE_URL:http://localhost:8000/predict_file/}
```

## AI Response Format

The AI service returns predictions in JSON format:
```json
{
  "prediction": {
    "species": "Plant species name",
    "stage": "Growth stage description",
    "confidence": 0.95,
    "height_ratio": 0.35
  }
}
```

## Future Enhancements

- Disease detection and diagnosis
- Nutrient deficiency identification
- Growth rate prediction
- Optimal harvesting time recommendation
- Environmental condition analysis