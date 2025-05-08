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

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        # Publisher for object position
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Nano model for faster inference

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

        # Timer for processing frames at ~30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

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
        results = self.model(frame, classes=[0])  # Class ID 0 = person

        # Initialize list for x-coordinates
        x_coordinates = []

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])  # Should be 0 (person)

                # Calculate center x-coordinate
                center_x = (x1 + x2) // 2
                x_coordinates.append(center_x)

                # Calculate area (for z in Point message)
                area = (x2 - x1) * (y2 - y1)

                # Draw bounding box and label
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Person: {conf:.2f}'
                cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Publish position for the first detected person
                if len(x_coordinates) == 1:  # Only publish for the first person
                    msg = Point()
                    msg.x = float(center_x)  # x-coordinate of person center
                    msg.y = 0.0  # Unused (flat-ground assumption)
                    msg.z = float(area)  # Area of bounding box
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