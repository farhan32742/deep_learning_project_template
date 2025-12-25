import os
import sys
from ultralytics import YOLO
from Deep_learning_projects.utils import log

class PredictionPipeline:
    def __init__(self):
        # Hardcoding the model path as per user instruction "my best.pt is on the model folder"
        # In a more robust setup, this could be in config.yaml
        self.model_path = os.path.join("model", "best(1).pt")
        
        if not os.path.exists(self.model_path):
             # Fallback or error if model doesn't exist. 
             # For now, we log a warning but let YOLO handle the error or download if it was a standard model.
             log.warning(f"Model not found at {self.model_path}. Please ensure 'best.pt' is in the 'model' folder.")

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            log.exception(f"Error loading model: {e}")
            raise e

    def predict(self, image_path):
        try:
            # Run inference
            results = self.model(image_path)
            
            # Process results (example: returning list of dicts)
            predictions = []
            for result in results:
                # result.boxes contains bounding boxes, confidence, class ids
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    # xyxy coordinates
                    coords = box.xyxy[0].tolist()
                    
                    predictions.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "box": coords
                    })
            return predictions

        except Exception as e:
            log.exception(f"Prediction failed: {e}")
            return {"error": str(e)}
