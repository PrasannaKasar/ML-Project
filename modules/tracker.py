# modules/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3):
        """
        DeepSORT Tracker
        :param max_age: frames to keep alive a track without detection
        :param n_init: number of frames before confirming a track
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, frame, detections):
        """
        Update tracker with new detections
        :param frame: current BGR frame (not used by DeepSORT but required)
        :param detections: list of dicts from YOLODetector
        :return: list of tracked objects:
                 (track_id, bbox, class_id, class_name, confidence)
        """
        # Convert detections to DeepSORT format: [x1, y1, x2, y2, conf, class_id]
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            cls = det['class_id']
            det_list.append([x1, y1, x2, y2, conf, cls])

        tracks = self.tracker.update_tracks(det_list, frame=frame)

        tracked_objects = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            ltrb = t.to_ltrb()  # left, top, right, bottom
            class_id = t.det_class if hasattr(t, 'det_class') else -1
            class_name = str(class_id)
            conf = t.det_conf if hasattr(t, 'det_conf') else 0
            tracked_objects.append((track_id, ltrb, class_id, class_name, conf))

        return tracked_objects

    def draw_tracks(self, frame, tracked_objects):
        """
        Draw tracked objects on frame
        :param frame: BGR image
        :param tracked_objects: list returned by update()
        :return: frame with tracks drawn
        """
        import cv2
        for track_id, bbox, class_id, class_name, conf in tracked_objects:
            x1, y1, x2, y2 = map(int, bbox)
            label = f"ID {track_id} {class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame
