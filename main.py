# main.py
from pipeline import DroneVisionPipeline
import pafy
import cv2
from vidgear.gears import CamGear

url = "https://www.youtube.com/watch?v=se1RDOPvA8Q"
stream = CamGear(source=url, stream_mode = True, logging=True).start() # YouTube Video URL as input
pipeline = DroneVisionPipeline(yolo_model="yolov5s.pt")
user_selected_id = None
# infinite loop
while True:
    
    frame = stream.read()
    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break

    output = pipeline.process_frame(frame, user_selected_id=user_selected_id)

    # After first frame, user selects target ID
    if user_selected_id is None and output["tracked_objects"]:
        for track_id, bbox, _, _, _ in output["tracked_objects"]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow("Select Target", frame)
        cv2.waitKey(1)
        user_selected_id = int(input("Enter target ID to track: "))

    # Optional: draw path
    if output["path"]:
        frame = pipeline.planner.draw_path(frame, output["path"])

    cv2.imshow("Drone Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
