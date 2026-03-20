#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load model OpenVINO
model = YOLO("/home/dung/ros2_ws/robot_vision/drink_detector_openvino_model")

points = []
roi_defined = False
roi_polygon = None

def mouse_callback(event, x, y, flags, param):
    global points, roi_defined, roi_polygon
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
        points.append((x, y))
        if len(points) == 3:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            x4 = x2 + (x3 - x1)
            y4 = y2 + (y3 - y1)
            roi_polygon = np.array([points[0], points[1], (x4, y4), points[2]])
            roi_defined = True

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Image, '/vision/detect_image', 10)

        cv2.namedWindow("Theo doi chai nuoc")
        cv2.setMouseCallback("Theo doi chai nuoc", mouse_callback)
        print("-> detect_openvino_ros.py READY: Tu dong dao truc Dung/Nam.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        global roi_defined, roi_polygon

        if roi_defined:
            cv2.polylines(frame, [roi_polygon], True, (255, 0, 0), 2)

        results = model(frame, conf=0.8, classes=[2])
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Chiều rộng và chiều cao box
                w_box = x2 - x1
                h_box = y2 - y1

                # Tâm giao điểm (P)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if h_box > w_box:
                    # TRƯỜNG HỢP CHAI ĐỨNG
                    half_val = (y2 - y1) / 2  
                    label = "DUNG"
                    line_end = (cx, y2) # Đường dọc xuống đáy
                else:
                    # TRƯỜNG HỢP CHAI NẰM
                    half_val = (x2 - x1) / 2  # Hoặc x2 - cx
                    label = "NAM"
                    line_end = (x2, cy) # Đường ngang sang phải

                is_inside = True
                if roi_defined:
                    if cv2.pointPolygonTest(roi_polygon, (cx, cy), False) < 0:
                        is_inside = False

                if is_inside:
                    # Vẽ Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Vẽ đường chéo trắng (X)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.line(frame, (x1, y2), (x2, y1), (255, 255, 255), 1)

                    # --- VẼ ĐƯỜNG ĐO (VÀNG) TỪ TÂM RA CẠNH THEO MODE ---
                    cv2.line(frame, (cx, cy), line_end, (0, 255, 255), 2)

                    # Vẽ chấm đỏ tại tâm
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Hiển thị thông số
                    display_text = f"Mode:{label} | Half:{int(half_val)}px"
                    cv2.putText(frame, display_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Theo doi chai nuoc", frame)
        cv2.waitKey(1)

        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.pub.publish(out_msg)
        

def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
