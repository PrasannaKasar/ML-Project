# modules/midas.py
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class MiDaSDepth:
    def __init__(self, model_type="DPT_Hybrid", device=None):
        """
        Lightweight MiDaS depth estimation
        :param model_type: "DPT_Hybrid", "DPT_Large", etc.
        :param device: "cuda" or "cpu"
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Load MiDaS model from torch.hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Transformation for input images
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, frame):
        """
        Estimate depth for the input frame
        :param frame: BGR image
        :return: depth map (H x W) normalized
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        # Normalize to 0-255 for visualization if needed
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        return depth_map

    def get_object_distance(self, bbox, depth_map):
        """
        Estimate distance of an object using depth map
        :param bbox: [x1, y1, x2, y2]
        :param depth_map: H x W depth array
        :return: approximate depth (median of bbox area)
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1]-1, x2), min(depth_map.shape[0]-1, y2)

        obj_depth = depth_map[y1:y2, x1:x2]
        if obj_depth.size == 0:
            return None
        # Median depth is robust to noise
        return float(np.median(obj_depth))
