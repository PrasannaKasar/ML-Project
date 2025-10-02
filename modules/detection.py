# modules/detection.py
import torch
import cv2

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4, device=None):
        """
        YOLO Detector for object detection
        :param model_name: Path or name of YOLOv8 model
        :param conf_threshold: Minimum confidence for detection
        :param device: "cuda" or "cpu" (auto-detect if None)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load YOLOv8 model using Ultralytics Hub
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name).to(self.device)
        self.model.conf = conf_threshold

    def detect(self, frame):
        """
        Run detection on a frame
        :param frame: BGR numpy image
        :return: List of detections, each as a dict:
                 {'bbox': [x1, y1, x2, y2], 'conf': confidence, 'class_id': int, 'class_name': str}
        """
        results = self.model(frame)
        detections = []

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            class_name = str(class_id)  # Replace with actual class names if available
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(conf),
                'class_id': class_id,
                'class_name': class_name
            })

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes with confidence on frame
        :param frame: BGR numpy image
        :param detections: list from detect()
        :return: frame with boxes drawn
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
