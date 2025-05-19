#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2
from ultralytics import YOLO
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import torch
from threading import Thread, Lock

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        # Publisher for object position
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        # Load YOLOv8 model
        self.yolo_model = YOLO('yolov8n.pt')  # Nano model for faster inference
        self.helmet_model = YOLO('hardhat.pt')  # Fine-tuned Hardhat model

        # GStreamer pipeline
        pipeline_str = (
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink'
        )

        # Initialize GStreamer
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.real_height = 1.7  # Real-world height of the person in meters
        self.focal_length = 800  # Approximate focal length in pixels (adjust for your camera)

        self.depth_map = None

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.frame_lock = Lock()

        # Timer for processing frames at ~30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

    def depth_thread(self, crop):
        if self.depth_map is not None:
            return self.depth_map

        with self.frame_lock:
            input_batch = self.midas_transforms(crop).to(self.device)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                self.depth_map = prediction.squeeze().cpu().numpy()
    
    def normalize_distance(self, distance, min_dist=2, max_dist=20):
        """
        Normalize the distance to a range from 0 (2m) to 10,000 (20m).
        Values less than 2m are clamped to 0.
        Values greater than 20m are clamped to 10000.
        """
        if distance <= min_dist:
            return 0
        elif distance >= max_dist:
            return 10000
        else:
            return int(((distance - min_dist) / (max_dist - min_dist)) * 10000)

    def on_timer(self):
        # Pull the latest frame from appsink
        sample = self.sink.emit('pull-sample')
        if not sample:
            self.get_logger().warn('No new frame available')
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            self.get_logger().warn('Failed to map buffer data')
            return

        # Convert buffer to numpy array
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Perform YOLOv8 inference (detect persons only)
        person_results = self.yolo_model(frame, classes=[0], conf=0.5)  # Class ID 0 = person
        helmet_results = self.helmet_model.predict(frame, conf=0.3)

        # Initialize list for x-coordinates
        x_coordinates = []

        # Process detection results
        for person_result in person_results:
            person_boxes = person_result.boxes
            for person_box in person_boxes:
                px1, py1, px2, py2 = map(int, person_box.xyxy[0])
                conf = person_box.conf[0]

                is_wearing_helmet = False
                for hr in helmet_results:
                    helmet_boxes = hr.boxes
                    for helmet_box in helmet_boxes:
                        helmet_class = int(helmet_box.cls[0])
                        if helmet_class == 1: # Class ID 1 = hardhat
                            hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 3)
                            if (hx1 > px1 and hx2 < px2):
                                is_wearing_helmet = True
                                break

                person_crop = frame[py1:py2, px1:px2]
                person_pixel_height = py2 - py1

                depth_t = Thread(target=self.depth_thread, args=(person_crop,))
                depth_t.start()
                depth_t.join()

                if person_pixel_height > 0 and self.depth_map is not None:
                    metric_distance = (self.real_height * self.focal_length) / person_pixel_height
                else:
                    metric_distance = 0

                normalized_distance = self.normalize_distance(metric_distance)

                center_x = (px1 + px2) // 2
                x_coordinates.append(center_x)

                cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (0, 255, 0), 2)
                label = f'Person: {conf:.2f}'
                label += " with Hardhat" if is_wearing_helmet else " without Hardhat"
                color = (0, 255, 0) if is_wearing_helmet else (0, 0, 255)

                cv2.putText(frame_bgr, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Distance: {metric_distance:.2f} m", 
                    (px1, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Publish position for the first detected person
                if len(x_coordinates) == 1:  # Only publish for the first person
                    msg = Point()
                    msg.x = float(center_x)  # x-coordinate of person center
                    msg.y = 0.0  # Unused (flat-ground assumption)
                    msg.z = float(normalized_distance)
                    self.position_pub.publish(msg)
                    self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')

        # Print x-coordinates
        for i, x in enumerate(x_coordinates):
            print(f"Person {i+1} x-coordinate: {x}")

        # Display the frame with detections
        cv2.imshow('YOLOv8 Person Detection', frame_bgr)
        cv2.waitKey(1)

    def destroy_node(self):
        # Cleanup GStreamer and OpenCV resources
        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()