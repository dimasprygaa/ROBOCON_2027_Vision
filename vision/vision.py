#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

# CONFIG
MODEL = "yolov8n-seg.pt"
CONF = 0.7

# REALSENSE
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# YOLO
YOLO_IMG_SIZE = 640

# OpenVINO device
# Bisa diganti:
# "CPU"
# "intel:gpu"
OPENVINO_DEVICE = "CPU"

# OUTPUT
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

# DEPTH
DEPTH_SCALE = 0.001

# NODE
class VisionNode(Node):

    def __init__(self):

        super().__init__("yolo_realsense_vision")

        # FPS
        self.fps = 0.0
        self.prev_time = time.perf_counter()

        # YOLO
        self.get_logger().info(
            "Loading YOLO model..."
        )

        model = YOLO(MODEL)

        # OPENVINO
        OV_MODEL = MODEL.replace(
            ".pt",
            "_openvino_model"
        )

        if not os.path.exists(OV_MODEL):

            self.get_logger().info(
                "Exporting YOLO to OpenVINO..."
            )

            model.export(
                format="openvino",
                imgsz=YOLO_IMG_SIZE
            )

        self.get_logger().info(
            f"Loading OpenVINO model: {OV_MODEL}"
        )

        self.model = YOLO(OV_MODEL)

        # REALSENSE PIPELINE
        self.get_logger().info(
            "Starting RealSense D455..."
        )

        self.pipeline = rs.pipeline()

        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.bgr8,
            CAMERA_FPS
        )

        self.config.enable_stream(
            rs.stream.depth,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.z16,
            CAMERA_FPS
        )

        # START CAMERA
        self.profile = self.pipeline.start(
            self.config
        )

        # DEPTH SCALE
        depth_sensor = (
            self.profile
            .get_device()
            .first_depth_sensor()
        )

        self.depth_scale = (
            depth_sensor.get_depth_scale()
        )

        self.get_logger().info(
            f"RealSense depth scale: "
            f"{self.depth_scale}"
        )

        # ALIGN DEPTH TO COLOR
        self.align = rs.align(
            rs.stream.color
        )

        # ROS PUBLISHER
        self.object_pub = self.create_publisher(
            String,
            "/vision/objects",
            10
        )

        # TIMER
        self.timer = self.create_timer(
            0.001,
            self.process
        )

        self.get_logger().info(
            "Vision node started"
        )

        self.get_logger().info(
            f"RealSense: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} "
            f"@ {CAMERA_FPS} FPS"
        )

        self.get_logger().info(
            f"Output: "
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
        )


    # GET DISTANCE FROM MASK
    def get_distance_from_mask(
        self,
        mask,
        depth
    ):

        if (
            mask.shape[0] != depth.shape[0]
            or
            mask.shape[1] != depth.shape[1]
        ):

            return None

        values = depth[
            mask > 0
        ]

        values = values[
            values > 0
        ]

        if len(values) == 0:

            return None

        low = np.percentile(
            values,
            10
        )

        high = np.percentile(
            values,
            90
        )

        filtered = values[
            (values >= low)
            &
            (values <= high)
        ]

        if len(filtered) == 0:

            return None

        median_depth = np.median(
            filtered
        )

        distance = (
            float(median_depth)
            *
            self.depth_scale
        )

        return distance

    # GET 3D POINT
    def get_3d_point(
        self,
        depth_frame,
        cx,
        cy
    ):

        try:

            depth_value = (
                depth_frame.get_distance(
                    cx,
                    cy
                )
            )

            if depth_value <= 0:

                return None

            depth_intrinsics = (
                depth_frame.profile
                .as_video_stream_profile()
                .intrinsics
            )

            point = rs.rs2_deproject_pixel_to_point(
                depth_intrinsics,
                [cx, cy],
                depth_value
            )

            return point

        except Exception:

            return None

    # PROCESS
    def process(self):

        try:

            frames = self.pipeline.wait_for_frames(
                timeout_ms=100
            )

        except Exception:

            return

        aligned_frames = self.align.process(
            frames
        )

        depth_frame = (
            aligned_frames.get_depth_frame()
        )

        color_frame = (
            aligned_frames.get_color_frame()
        )

        if not depth_frame or not color_frame:

            return

        frame = np.asanyarray(
            color_frame.get_data()
        )

        depth = np.asanyarray(
            depth_frame.get_data()
        )


        # RESIZE RGB
        frame = cv2.resize(

            frame,

            (
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT
            ),

            interpolation=cv2.INTER_AREA
        )

        # RESIZE DEPTH
        depth = cv2.resize(

            depth,

            (
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT
            ),

            interpolation=cv2.INTER_NEAREST
        )

        results = self.model.predict(

            source=frame,

            imgsz=YOLO_IMG_SIZE,

            conf=CONF,

            verbose=False,

            device=OPENVINO_DEVICE

        )

        if results is None:

            return

        if len(results) == 0:

            return

        result = results[0]

        annotated = result.plot()

        objects = []

        # DETECTIONS
        if result.boxes is not None:

            for i, box in enumerate(
                result.boxes
            ):

                cls_id = int(
                    box.cls[0]
                )

                class_name = (
                    self.model.names[
                        cls_id
                    ]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cx = (
                    x1 + x2
                ) // 2

                cy = (
                    y1 + y2
                ) // 2

                # DEPTH

                distance = None

                if result.masks is not None:

                    try:

                        mask = (
                            result
                            .masks
                            .data[i]
                            .cpu()
                            .numpy()
                        )

                        # Resize mask
                        mask = cv2.resize(

                            mask,

                            (
                                OUTPUT_WIDTH,
                                OUTPUT_HEIGHT
                            ),

                            interpolation=cv2.INTER_NEAREST
                        )

                        mask = (
                            mask > 0.5
                        )

                        distance = (
                            self.get_distance_from_mask(
                                mask,
                                depth
                            )
                        )

                    except Exception as e:

                        self.get_logger().error(
                            f"Mask error: {e}"
                        )

                # 3D POSITION

                point_3d = None

                # Karena frame sudah di-resize,
                # scale kembali ke resolusi kamera

                original_cx = int(
                    cx
                    *
                    CAMERA_WIDTH
                    /
                    OUTPUT_WIDTH
                )

                original_cy = int(
                    cy
                    *
                    CAMERA_HEIGHT
                    /
                    OUTPUT_HEIGHT
                )

                point_3d = self.get_3d_point(

                    depth_frame,

                    original_cx,

                    original_cy

                )

                cv2.circle(

                    annotated,

                    (cx, cy),

                    6,

                    (0, 0, 255),

                    -1
                )

                if distance is not None:

                    distance_text = (
                        f"{distance:.2f} m"
                    )

                else:

                    distance_text = (
                        "Depth N/A"
                    )

                text = (

                    f"{class_name} "
                    f"{confidence:.2f} "
                    f"| {distance_text}"

                )

                cv2.putText(

                    annotated,

                    text,

                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2,

                    cv2.LINE_AA
                )

                if point_3d is not None:

                    X = point_3d[0]
                    Y = point_3d[1]
                    Z = point_3d[2]

                    print(

                        f"{class_name} | "
                        f"{confidence:.2f} | "
                        f"Center=({cx},{cy}) | "
                        f"Distance={distance if distance is not None else 0:.2f} m | "
                        f"XYZ=({X:.2f}, "
                        f"{Y:.2f}, "
                        f"{Z:.2f}) m",

                        flush=True
                    )

                else:

                    print(

                        f"{class_name} | "
                        f"{confidence:.2f} | "
                        f"Center=({cx},{cy}) | "
                        f"Distance=N/A",

                        flush=True
                    )

                object_data = {

                    "class": class_name,

                    "confidence": round(
                        confidence,
                        3
                    ),

                    "center_x": cx,

                    "center_y": cy,

                    "distance": (
                        round(distance, 3)
                        if distance is not None
                        else None
                    ),

                    "x": (
                        round(float(point_3d[0]), 3)
                        if point_3d is not None
                        else None
                    ),

                    "y": (
                        round(float(point_3d[1]), 3)
                        if point_3d is not None
                        else None
                    ),

                    "z": (
                        round(float(point_3d[2]), 3)
                        if point_3d is not None
                        else None
                    )
                }

                objects.append(
                    object_data
                )

        # PUBLISH ROS DATA

        msg = String()

        msg.data = str(
            objects
        )

        self.object_pub.publish(
            msg
        )

        # FPS
        current_time = time.perf_counter()

        elapsed_time = (
            current_time
            -
            self.prev_time
        )

        if elapsed_time > 0:

            instant_fps = (
                1.0
                /
                elapsed_time
            )

            # Smooth FPS
            self.fps = (
                0.9 * self.fps
                +
                0.1 * instant_fps
            )

        self.prev_time = current_time

        cv2.putText(

            annotated,

            f"FPS: {self.fps:.1f}",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2,

            cv2.LINE_AA
        )

        cv2.imshow(

            "YOLO11 Seg + RealSense D455",

            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            rclpy.shutdown()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = VisionNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as e:

        print(
            f"[ERROR] {e}"
        )

    finally:

        if node is not None:

            try:

                node.pipeline.stop()

            except Exception:

                pass

            node.destroy_node()

        cv2.destroyAllWindows()

        if rclpy.ok():

            rclpy.shutdown()

if __name__ == "__main__":

    main()