import time
import svgwrite
import numpy as np
import scipy.ndimage
import cv2
import serial
import json
from pose_engine import PoseEngine, EDGES, BODYPIX_PARTS

# ... [Previous color mappings and helper functions remain the same] ...

class PoseTracker:
    def __init__(self, engine, anonymize=True, bodyparts=True):
        # ... [Previous initialization code remains the same] ...
        self.tracked_wrist = None

    def __call__(self, image, svg_canvas):
        start_time = time.monotonic()
        inference_time, poses, heatmap, bodyparts = self.engine.DetectPosesInImage(image)

        # ... [Previous heatmap and background processing code remains the same] ...

        output_image = self.background_image + rgb_heatmap
        int_img = np.uint8(np.clip(output_image,0,255))

        # ... [Previous timing and text drawing code remains the same] ...

        self.tracked_wrist = None  # Reset tracked wrist for each frame
        for pose in poses:
            draw_pose(svg_canvas, pose)
            
            if 'right_wrist' in pose.keypoints:
                wrist = pose.keypoints['right_wrist']
                if wrist.score > 0.2:  # Confidence threshold
                    wrist_x = int(wrist.yx[1])
                    wrist_y = int(wrist.yx[0])

                    # Only track the first detected right wrist
                    if self.tracked_wrist is None:
                        self.tracked_wrist = (wrist_x, wrist_y)

                        # Draw a colored box around the wrist
                        box_size = 40
                        cv2.rectangle(int_img, 
                                      (wrist_x - box_size//2, wrist_y - box_size//2), 
                                      (wrist_x + box_size//2, wrist_y + box_size//2), 
                                      (0, 255, 0), 2)
                        
                        # Label the wrist as "1st"
                        cv2.putText(int_img, "1st", (wrist_x - 20, wrist_y - 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Calculate error from the center of the frame
                        frame_center_x = image.shape[1] // 2
                        frame_center_y = image.shape[0] // 2
                        error_x = wrist_x - frame_center_x
                        error_y = wrist_y - frame_center_y

                        # Pan logic
                        if error_x > 30:  # Wrist is to the right
                            self.pan_angle -= 1
                            if self.pan_angle < -180:
                                self.pan_angle = -180
                        elif error_x < -30:  # Wrist is to the left
                            self.pan_angle += 1
                            if self.pan_angle > 180:
                                self.pan_angle = 180

                        # Tilt logic
                        if error_y > 30:  # Wrist is below the center
                            self.tilt_angle += 1
                            if self.tilt_angle > 90:
                                self.tilt_angle = 90
                        elif error_y < -30:  # Wrist is above the center
                            self.tilt_angle -= 1
                            if self.tilt_angle < -30:
                                self.tilt_angle = -30

                        # Prepare command
                        command = {
                            "T": 133,
                            "X": self.pan_angle,
                            "Y": self.tilt_angle,
                            "SPD": 0,
                            "ACC": 0
                        }

                        json_command = json.dumps(command)
                        print("Sending command:", json_command)

                        try:
                            self.ser.write((json_command + '\n').encode('utf-8'))
                            print("Command sent successfully")
                        except serial.SerialException as e:
                            print(f'Failed to send command: {e}')

        print(text_line)
        return int_img

def main():
    # ... [Previous main function code remains the same] ...

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        svg_canvas = svgwrite.Drawing('', size=(640, 480))
        output_frame = tracker(frame, svg_canvas)

        # Convert SVG to NumPy array and overlay on the output frame
        svg_string = svg_canvas.tostring()
        svg_image = cv2.imdecode(np.frombuffer(svg_string.encode(), np.uint8), cv2.IMREAD_UNCHANGED)
        if svg_image is not None and svg_image.shape[2] == 4:  # Check if alpha channel exists
            alpha = svg_image[:, :, 3] / 255.0
            rgb = svg_image[:, :, :3]
            output_frame = output_frame * (1 - alpha[:, :, np.newaxis]) + rgb * alpha[:, :, np.newaxis]
        
        output_frame = np.uint8(output_frame)

        cv2.imshow('PoseNet Tracking', output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.ser.close()

if __name__ == '__main__':
    main()