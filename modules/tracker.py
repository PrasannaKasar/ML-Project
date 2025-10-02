# modules/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_cosine_distance=0.5, nn_budget=100):
        """
        Initialize the tracker with parameters
        :param max_cosine_distance: max cosine distance for matching
        :param nn_budget: budget for nearest neighbor (memory) in DeepSORT
        """
        self.tracker = DeepSort(max_cosine_distance=max_cosine_distance, nn_budget=nn_budget)

    def update(self, frame, detections):
        """
        Update the tracker with new detections
        :param frame: The current frame (for visualization)
        :param detections: List of detections in the format [x_min, y_min, x_max, y_max, confidence, class_id]
        :return: List of tracked objects
        """
        # Ensure the detections are in the correct format
        formatted_detections = []
        for det in detections:
            # Check if the detection has 6 elements
            if len(det) == 6:
                x_min, y_min, x_max, y_max, confidence, class_id = det
                # Add the detection in [x_min, y_min, x_max, y_max, confidence, class_id] format
                formatted_detections.append([x_min, y_min, x_max, y_max, confidence, class_id])
            else:
                print(f"Skipping invalid detection: {det}")

        # Ensure we have valid detections
        if len(formatted_detections) > 0:
            # Update the DeepSORT tracker
            tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        else:
            tracks = []

        return tracks

    def draw_tracks(self, frame, tracks):
        """
        Draw the tracks on the frame
        :param frame: The current frame
        :param tracks: The list of tracked objects
        :return: Frame with drawn tracks
        """
        for track in tracks:
            # Draw a rectangle for each track
            x1, y1, x2, y2 = track[0]  # Get the bounding box coordinates
            track_id = track[1]  # Get the track ID

            # Draw the bounding box and track ID
            color = (0, 255, 0)  # Green color for tracks
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            frame = cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame
