import time
import svgwrite
import numpy as np
import scipy.ndimage
import cv2
import serial
import json
from pose_engine import PoseEngine, EDGES, BODYPIX_PARTS

# Color mapping for bodyparts
RED_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right" in v]
GREEN_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "hand" in v or "torso" in v]
BLUE_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "leg" in v or "arm" in v or "face" in v or "hand" in v]

def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))

def draw_pose(dwg, pose, color='blue', threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        dwg.add(dwg.circle(center=(int(keypoint.yx[1]), int(keypoint.yx[0])), r=5,
                           fill='cyan', stroke=color))
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))

class PoseTracker:
    def __init__(self, engine, anonymize=True, bodyparts=True):
        self.engine = engine
        self.anonymize = anonymize
        self.bodyparts = bodyparts
        self.background_image = None
        self.last_time = time.monotonic()
        self.frames = 0
        self.sum_fps = 0
        self.sum_process_time = 0
        self.sum_inference_time = 0
        
        # Initialize serial connection
        self.serial_port = '/dev/ttymxc3'
        self.baud_rate = 115200
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        
        # Pan and tilt angles initialized
        self.pan_angle = 0
        self.tilt_angle = 0

    def __call__(self, image, svg_canvas):
        start_time = time.monotonic()
        inference_time, poses, heatmap, bodyparts = self.engine.DetectPosesInImage(image)

        def clip_heatmap(heatmap, v0, v1):
            a = v0 / (v0 - v1)
            b = 1.0 / (v1 - v0)
            return np.clip(a + b * heatmap, 0.0, 1.0)

        # clip heatmap to create a mask
        heatmap = clip_heatmap(heatmap, -1.0, 1.0)

        if self.bodyparts:
            rgb_heatmap = np.dstack([
                heatmap*(np.sum(bodyparts[:,:,RED_BODYPARTS], axis=2)-0.5)*100,
                heatmap*(np.sum(bodyparts[:,:,GREEN_BODYPARTS], axis=2)-0.5)*100,
                heatmap*(np.sum(bodyparts[:,:,BLUE_BODYPARTS], axis=2)-0.5)*100,
            ])
        else:
            rgb_heatmap = np.dstack([heatmap[:,:]*100]*3)
            rgb_heatmap[:,:,1:] = 0 # make it red

        rgb_heatmap = 155*np.clip(rgb_heatmap, 0, 1)
        rescale_factor = [
            image.shape[0]/heatmap.shape[0],
            image.shape[1]/heatmap.shape[1],
            1
        ]

        rgb_heatmap = scipy.ndimage.zoom(rgb_heatmap, rescale_factor, order=0)

        if self.anonymize:
            if self.background_image is None:
                self.background_image = np.float32(np.zeros_like(image))
            # Estimate instantaneous background
            mask = np.clip(np.sum(rgb_heatmap, axis=2), 0, 1)[:,:,np.newaxis]
            background_estimate = (self.background_image*mask + image*(1.0-mask))

            # Mix into continuous estimate with decay
            ratio = 1/max(1,self.frames/2.0)
            self.background_image = self.background_image*(1.0-ratio) + ratio*background_estimate
        else:
            self.background_image = image

        output_image = self.background_image + rgb_heatmap
        int_img = np.uint8(np.clip(output_image,0,255))

        end_time = time.monotonic()

        self.frames += 1
        self.sum_fps += 1.0 / (end_time - self.last_time)
        self.sum_process_time += 1000 * (end_time - start_time) - inference_time
        self.sum_inference_time += inference_time
        self.last_time = end_time
        text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
            self.sum_inference_time / self.frames,
            self.sum_process_time / self.frames,
            self.sum_fps / self.frames,
            len(poses)
        )

        shadow_text(svg_canvas, 10, 20, text_line)
        for pose in poses:
            draw_pose(svg_canvas, pose)
            
            # Track right wrist for pan-tilt
            if 'right_wrist' in pose.keypoints:
                wrist = pose.keypoints['right_wrist']
                if wrist.score > 0.2:  # Confidence threshold
                    wrist_x = int(wrist.yx[1])
                    wrist_y = int(wrist.yx[0])

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
    model = 'models/bodypix_mobilenet_v1_075_640_480_16_quant_decoder_edgetpu.tflite'
    print('Model: {}'.format(model))

    engine = PoseEngine(model)
    inference_size = (engine.image_width, engine.image_height)
    print('Inference size: {}'.format(inference_size))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = PoseTracker(engine, anonymize=False, bodyparts=True)

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
        output_frame = cv2.addWeighted(output_frame, 1, svg_image, 0.5, 0)

        cv2.imshow('PoseNet Tracking', output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.ser.close()

if __name__ == '__main__':
    main()