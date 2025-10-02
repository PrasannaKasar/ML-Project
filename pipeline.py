# pipeline.py
from modules.detection import YOLODetector
from modules.tracker import ObjectTracker
from modules.filter import select_target
from modules.midas import MiDaSDepth
from modules.path_plan import AStarPlanner

class DroneVisionPipeline:
    def __init__(self, yolo_model="yolov5s.pt", obstacle_threshold=0.2):
        # Modules
        self.detector = YOLODetector(model_name=yolo_model)
        self.tracker = ObjectTracker()
        self.depth_estimator = MiDaSDepth()
        self.planner = AStarPlanner(obstacle_threshold=obstacle_threshold)

        # Target tracking
        self.target_id = None

    def process_frame(self, frame, user_selected_id=None):
        """
        Process a single frame through the pipeline
        :param frame: BGR image
        :param user_selected_id: ID selected by user on first frame
        :return: dict with detections, tracked_objects, target, depth_map, path
        """
        # 1️⃣ Detection
        detections = self.detector.detect(frame)

        # 2️⃣ Tracking
        tracked_objects = self.tracker.update(frame, detections)

        # 3️⃣ Set target ID (user selection for first frame)
        if self.target_id is None and user_selected_id is not None:
            self.target_id = user_selected_id

        # 4️⃣ Filter tracked objects to get target
        target_objects = select_target(tracked_objects, target_id=self.target_id)

        # 5️⃣ Depth estimation
        depth_map = self.depth_estimator.estimate_depth(frame)

        # 6️⃣ Compute path for target
        path = []
        if target_objects:
            track_id, bbox, class_id, class_name, conf = target_objects[0]
            # Use bbox center as goal
            x1, y1, x2, y2 = bbox
            goal = ((x1 + x2)//2, (y1 + y2)//2)
            # Start can be frame center (or drone center in real system)
            start = (frame.shape[1]//2, frame.shape[0]//2)
            path = self.planner.plan(depth_map, start, goal)

        return {
            "detections": detections,
            "tracked_objects": tracked_objects,
            "target_objects": target_objects,
            "depth_map": depth_map,
            "path": path
        }
