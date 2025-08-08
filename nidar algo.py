#!/usr/bin/env python3
"""
DISASTER RESPONSE DRONE SWARM AI SYSTEM
Version: 3.0
Developed for: Pixhawk 4 + Raspberry Pi 5
Features:
- Multi-agent swarm intelligence
- AI-driven sensor calibration
- Real-time 3D mapping
- Survivor detection and rescue coordination
- Distributed computing architecture
- Fault-tolerant communication
"""

import os
import sys
import time
import json
import math
import queue
import socket
import threading
import logging
import argparse
import numpy as np
import cv2
import torch
import tensorflow as tf
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Import drone-specific libraries
from pymavlink import mavutil
from dronekit import connect, Vehicle, VehicleMode, LocationGlobalRelative
from ultralytics import YOLO
from keras.models import Model, Sequential, load_model
from keras.layers import (LSTM, Dense, Conv2D, MaxPooling2D, 
                          Flatten, Dropout, Input, concatenate)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drone_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DroneAI")

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================
class DroneRole(Enum):
    SCOUT = auto()
    DELIVERY = auto()
    COMMAND = auto()

class MissionState(Enum):
    PREFLIGHT = auto()
    TAKEOFF = auto()
    SEARCH = auto()
    RESCUE = auto()
    DELIVERY = auto()
    RETURN = auto()
    LANDING = auto()
    EMERGENCY = auto()

class SensorType(Enum):
    GPS = auto()
    IMU = auto()
    LIDAR = auto()
    RGB_CAM = auto()
    THERMAL_CAM = auto()
    GAS_SENSOR = auto()
    BAROMETER = auto()
    
class TerrainMapper:
    def __init__(self):
        self.occupancy_grid = None
        self.resolution = 0.5  # meters per cell
        self.map_size = (500, 500)  # cells
    
    def update_map(self, position, sensor_data):
        # Convert position to grid coordinates
        grid_x = int(position[0] / self.resolution)
        grid_y = int(position[1] / self.resolution)
        
        # Update grid based on sensor readings
        for angle, distance in sensor_data['lidar']:
            # Calculate obstacle position
            obs_x = grid_x + int(distance * math.cos(angle) / self.resolution)
            obs_y = grid_y + int(distance * math.sin(angle) / self.resolution)
            
            # Update occupancy grid
            if 0 <= obs_x < self.map_size[0] and 0 <= obs_y < self.map_size[1]:
                self.occupancy_grid[obs_x, obs_y] = 1
        
        # Update free space
        for x in range(grid_x-10, grid_x+10):
            for y in range(grid_y-10, grid_y+10):
                if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                    if self.occupancy_grid[x, y] != 1:
                        self.occupancy_grid[x, y] = 0

class SwarmOptimizer:
    def __init__(self, swarm_size):
        self.swarm_size = swarm_size
        self.positions = np.zeros((swarm_size, 3))
        self.velocities = np.zeros((swarm_size, 3))
        self.best_positions = np.copy(self.positions)
        self.global_best = None
    
    def update(self, fitness_function):
        # Particle Swarm Optimization
        w = 0.729  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        
        for i in range(self.swarm_size):
            # Update velocity
            r1 = np.random.random(3)
            r2 = np.random.random(3)
            cognitive = c1 * r1 * (self.best_positions[i] - self.positions[i])
            social = c2 * r2 * (self.global_best - self.positions[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social
            
            # Update position
            self.positions[i] += self.velocities[i]
            
            # Evaluate fitness
            current_fitness = fitness_function(self.positions[i])
            if current_fitness < fitness_function(self.best_positions[i]):
                self.best_positions[i] = self.positions[i]
            
            if current_fitness < fitness_function(self.global_best):
                self.global_best = self.positions[i]

class DisasterSimulation:
    def __init__(self, area_size=(1000, 1000)):
        self.area_size = area_size
        self.survivors = []
        self.hazards = []
        self.obstacles = []
        self.weather = "CLEAR"
    
    def generate_scenario(self, num_survivors=10, num_hazards=5, num_obstacles=20):
        # Generate random survivors
        self.survivors = [
            (np.random.uniform(0, self.area_size[0]),
             np.random.uniform(0, self.area_size[1]),
             np.random.uniform(0, 5))  # Elevation
            for _ in range(num_survivors)
        ]
        
        # Generate hazards (fires, floods, etc.)
        self.hazards = [
            {'type': np.random.choice(["FIRE", "FLOOD", "GAS_LEAK"]),
             'position': (np.random.uniform(0, self.area_size[0]),
                          np.random.uniform(0, self.area_size[1]),
                          np.random.uniform(0, 5)),
             'intensity': np.random.uniform(0.5, 1.0)}
            for _ in range(num_hazards)
        ]
        
        # Generate obstacles (buildings, debris)
        self.obstacles = [
            {'position': (np.random.uniform(0, self.area_size[0]),
                          np.random.uniform(0, self.area_size[1]),
                          np.random.uniform(0, 20)),
             'size': np.random.uniform(5, 50)}
            for _ in range(num_obstacles)
        ]
    
    def update(self, drones):
        # Simulate environmental changes
        for hazard in self.hazards:
            if hazard['type'] == "FIRE":
                hazard['intensity'] += 0.01
                if hazard['intensity'] > 1.0:
                    # Spread fire
                    new_fire = {
                        'type': "FIRE",
                        'position': (
                            hazard['position'][0] + np.random.uniform(-50, 50),
                            hazard['position'][1] + np.random.uniform(-50, 50),
                            hazard['position'][2]
                        ),
                        'intensity': 0.3
                    }
                    self.hazards.append(new_fire)
                    hazard['intensity'] = 0.7
        
        # Update survivor states
        for survivor in self.survivors:
            # Random movement
            survivor[0] += np.random.uniform(-1, 1)
            survivor[1] += np.random.uniform(-1, 1)

DRONE_CONFIG = {
    "max_altitude": 120,  # meters
    "safe_distance": 5.0,  # meters
    "battery_threshold": 25.0,  # percentage
    "communication_range": 1000,  # meters
    "swarm_update_interval": 1.0,  # seconds
    "calibration_interval": 10.0,  # seconds
    "emergency_timeout": 30.0  # seconds
}

SENSOR_CONFIG = {
    "gps_accuracy_threshold": 2.0,  # meters
    "imu_sample_rate": 100,  # Hz
    "lidar_range": 30.0,  # meters
    "camera_resolution": (1920, 1080),
    "thermal_range": (-20, 400)  # Celsius
}

AI_MODEL_PATHS = {
    "object_detection": "models/yolov8_disaster_v3.pt",
    "thermal_analysis": "models/thermal_unet_v2.h5",
    "sensor_calibration": "models/lstm_calibrator_v4.h5",
    "path_planning": "models/rl_pathfinder_v3.h5",
    "swarm_coordination": "models/swarm_gnn_v2.pt"
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class DroneState:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # lat, lon, alt
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s
    attitude: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    battery: float = 100.0
    mode: VehicleMode = VehicleMode("GUIDED")
    status: MissionState = MissionState.PREFLIGHT
    health: Dict[str, float] = field(default_factory=dict)
    last_update: float = time.time()

@dataclass
class SwarmMember:
    drone_id: int
    role: DroneRole
    state: DroneState
    capabilities: List[str]
    ip_address: str
    last_seen: float = time.time()

@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    position: Tuple[float, float, float]
    bbox: Tuple[float, float, float, float]
    depth: float
    temperature: Optional[float] = None
    timestamp: float = time.time()

@dataclass
class MissionTask:
    task_id: int
    task_type: str
    priority: int
    location: Tuple[float, float, float]
    assigned_to: int = -1
    status: str = "PENDING"
    created_at: float = time.time()

# =============================================================================
# SENSOR INTERFACE MODULE
# =============================================================================
class SensorInterface:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.sensor_data = {st: None for st in SensorType}
        self.sensor_thread = threading.Thread(target=self._sensor_update_loop)
        self.running = True
        self.calibration_params = {}
        self._initialize_sensors()
        
    def _initialize_sensors(self):
        """Initialize all connected sensors"""
        # GPS initialization
        while not self.vehicle.gps_0:
            logger.warning("Waiting for GPS fix...")
            time.sleep(1)
            
        # IMU calibration
        self._calibrate_imu()
        
        # Camera initialization
        self._initialize_cameras()
        
        # Start sensor update thread
        self.sensor_thread.start()
        logger.info("Sensor interface initialized")
    
    def _calibrate_imu(self):
        """Perform initial IMU calibration"""
        logger.info("Starting IMU calibration...")
        self.vehicle.send_mavlink(
            self.vehicle.message_factory.command_long_encode(
                0, 0,  # target_system, target_component
                mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,  # command
                0,  # confirmation
                1, 0, 0, 0, 0, 0, 0  # parameters (1 = calibrate gyro)
            )
        )
        time.sleep(15)  # Wait for calibration to complete
        logger.info("IMU calibration complete")
    
    def _initialize_cameras(self):
        """Initialize all camera sensors"""
        # RGB camera
        self.rgb_cam = cv2.VideoCapture(0)
        if not self.rgb_cam.isOpened():
            logger.error("Failed to open RGB camera")
        else:
            self.rgb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, SENSOR_CONFIG["camera_resolution"][0])
            self.rgb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, SENSOR_CONFIG["camera_resolution"][1])
            logger.info("RGB camera initialized")
        
        # Thermal camera (simulated for this example)
        self.thermal_cam = None  # In real system, this would be a FLIR Lepton interface
        
    def _sensor_update_loop(self):
        """Continuous sensor data collection thread"""
        while self.running:
            try:
                # Update GPS data
                if self.vehicle.gps_0:
                    self.sensor_data[SensorType.GPS] = (
                        self.vehicle.location.global_frame.lat,
                        self.vehicle.location.global_frame.lon,
                        self.vehicle.location.global_frame.alt
                    )
                
                # Update IMU data
                self.sensor_data[SensorType.IMU] = (
                    self.vehicle.attitude.roll,
                    self.vehicle.attitude.pitch,
                    self.vehicle.attitude.yaw,
                    self.vehicle.velocity[0],
                    self.vehicle.velocity[1],
                    self.vehicle.velocity[2],
                    self.vehicle.raw_imu.xacc,
                    self.vehicle.raw_imu.yacc,
                    self.vehicle.raw_imu.zacc
                )
                
                # Update other sensors (simulated)
                self.sensor_data[SensorType.LIDAR] = np.random.uniform(0.1, SENSOR_CONFIG["lidar_range"])
                self.sensor_data[SensorType.GAS_SENSOR] = np.random.uniform(0, 100)  # ppm
                self.sensor_data[SensorType.BAROMETER] = self.vehicle.location.global_frame.alt
                
                time.sleep(0.01)  # Approx 100Hz update rate
                
            except Exception as e:
                logger.error(f"Sensor update error: {e}")
                time.sleep(1)
    
    def get_sensor_data(self, sensor_type: SensorType):
        """Get current sensor reading"""
        return self.sensor_data.get(sensor_type, None)
    
    def capture_rgb_image(self):
        """Capture an image from the RGB camera"""
        if self.rgb_cam and self.rgb_cam.isOpened():
            ret, frame = self.rgb_cam.read()
            if ret:
                return frame
        return None
    
    def capture_thermal_image(self):
        """Capture an image from the thermal camera (simulated)"""
        # In a real system, this would interface with a FLIR camera
        return np.random.rand(160, 120) * 100  # Simulated 160x120 thermal image
    
    def apply_calibration(self, params: Dict[str, float]):
        """Apply calibration parameters to sensors"""
        self.calibration_params = params
        # In a real system, this would adjust sensor readings
        logger.info(f"Applied new calibration parameters: {params}")
    
    def shutdown(self):
        """Cleanup sensor resources"""
        self.running = False
        if self.rgb_cam:
            self.rgb_cam.release()
        logger.info("Sensors shutdown")

# =============================================================================
# AI PERCEPTION MODULE
# =============================================================================
class AIPerception:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.object_model = self._load_model(AI_MODEL_PATHS["object_detection"])
        self.thermal_model = self._load_model(AI_MODEL_PATHS["thermal_analysis"])
        self.class_map = self._load_class_map()
        self.tracker = ObjectTracker()
        self.detection_history = []
        logger.info(f"Drone {drone_id} perception system initialized")
    
    def _load_model(self, model_path: str):
        """Load an AI model from file"""
        try:
            if model_path.endswith('.pt'):
                model = YOLO(model_path)
                logger.info(f"Loaded YOLO model: {model_path}")
                return model
            elif model_path.endswith('.h5'):
                model = load_model(model_path)
                logger.info(f"Loaded Keras model: {model_path}")
                return model
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return None
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None
    
    def _load_class_map(self):
        """Load class ID to name mapping"""
        return {
            0: "survivor",
            1: "fire",
            2: "debris",
            3: "landing_zone",
            4: "flood_water",
            5: "structural_damage",
            6: "medical_kit",
            7: "hazardous_material"
        }
    
    def detect_objects(self, rgb_frame, thermal_frame=None):
        """Perform object detection on input frame"""
        detections = []
        
        # Run object detection
        results = self.object_model.predict(
            source=rgb_frame, 
            conf=0.7, 
            iou=0.5,
            augment=True,
            verbose=False
        )
        
        # Process detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box
                class_name = self.class_map.get(int(cls_id), "unknown")
                
                # Estimate depth (simulated - in real system use LiDAR)
                depth = self._estimate_depth(box, rgb_frame.shape)
                
                # Thermal analysis if available
                temperature = None
                if thermal_frame is not None and class_name == "survivor":
                    temperature = self._analyze_thermal(thermal_frame, box)
                
                detection = DetectionResult(
                    class_name=class_name,
                    confidence=float(conf),
                    position=(0, 0, 0),  # Will be filled later
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    depth=float(depth),
                    temperature=temperature
                )
                
                detections.append(detection)
        
        # Track objects across frames
        tracked_detections = self.tracker.update(detections)
        self.detection_history.extend(tracked_detections)
        return tracked_detections
    
    def _estimate_depth(self, bbox, frame_shape):
        """Estimate object depth based on bounding box size"""
        # Simplified model - in real system use LiDAR or stereo vision
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame_shape[0] * frame_shape[1]
        relative_size = bbox_area / frame_area
        
        # Empirical model: closer objects appear larger
        if relative_size > 0.3:
            return np.random.uniform(1.0, 5.0)  # 1-5 meters
        elif relative_size > 0.1:
            return np.random.uniform(5.0, 15.0)  # 5-15 meters
        else:
            return np.random.uniform(15.0, SENSOR_CONFIG["lidar_range"])  # 15-30 meters
    
    def _analyze_thermal(self, thermal_frame, bbox):
        """Analyze thermal signature in region of interest"""
        try:
            # Convert bbox to thermal image coordinates (simplified)
            x1, y1, x2, y2 = bbox
            x1_t = int(x1 * thermal_frame.shape[1] / SENSOR_CONFIG["camera_resolution"][0])
            y1_t = int(y1 * thermal_frame.shape[0] / SENSOR_CONFIG["camera_resolution"][1])
            x2_t = int(x2 * thermal_frame.shape[1] / SENSOR_CONFIG["camera_resolution"][0])
            y2_t = int(y2 * thermal_frame.shape[0] / SENSOR_CONFIG["camera_resolution"][1])
            
            # Extract region of interest
            roi = thermal_frame[y1_t:y2_t, x1_t:x2_t]
            if roi.size == 0:
                return None
                
            # Calculate average temperature
            return np.mean(roi)
        except Exception as e:
            logger.error(f"Thermal analysis failed: {e}")
            return None
    
    def generate_thermal_map(self, thermal_frame):
        """Generate a visual thermal map"""
        try:
            # Normalize to 0-255 range
            normalized = (thermal_frame - SENSOR_CONFIG["thermal_range"][0]) / (
                SENSOR_CONFIG["thermal_range"][1] - SENSOR_CONFIG["thermal_range"][0]) * 255
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # Apply color map
            return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        except Exception as e:
            logger.error(f"Thermal map generation failed: {e}")
            return None
    
    def detect_obstacles(self, depth_data):
        """Detect obstacles from depth data"""
        # This would use point cloud processing in a real system
        obstacles = []
        if depth_data < 10.0:  # If something closer than 10m
            obstacles.append({
                "direction": "front",
                "distance": depth_data,
                "severity": "high" if depth_data < 5.0 else "medium"
            })
        return obstacles

# =============================================================================
# OBJECT TRACKING MODULE
# =============================================================================
class ObjectTracker:
    def __init__(self, max_age=5):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Simple tracking implementation - in real system use DeepSORT or similar
        updated_detections = []
        
        # Update existing tracks
        for track_id, track in list(self.tracks.items()):
            matched = False
            for det in detections:
                if self._iou(track["bbox"], det.bbox) > 0.5:
                    # Update track
                    track["bbox"] = det.bbox
                    track["last_seen"] = time.time()
                    det.track_id = track_id
                    updated_detections.append(det)
                    matched = True
                    break
            
            # Remove old tracks
            if not matched:
                if time.time() - track["last_seen"] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks
        for det in detections:
            if not hasattr(det, 'track_id') or det.track_id is None:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    "bbox": det.bbox,
                    "class": det.class_name,
                    "first_seen": time.time(),
                    "last_seen": time.time()
                }
                self.next_id += 1
                updated_detections.append(det)
        
        return updated_detections
    
    def _iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

# =============================================================================
# CALIBRATION AI MODULE
# =============================================================================
class CalibrationAI:
    def __init__(self):
        self.model = self._build_model()
        self.sensor_buffer = np.zeros((100, 10))  # Stores last 100 sensor readings
        self.calibration_history = []
        self.last_calibration_time = 0
    
    def _build_model(self):
        """Build LSTM-based calibration model"""
        model = Sequential([
            LSTM(128, input_shape=(100, 10), return_sequences=True),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(6, activation='linear')  # Output: 6 calibration offsets
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def update_sensor_data(self, sensor_data):
        """Update sensor buffer with new readings"""
        # Rotate buffer and add new data
        self.sensor_buffer = np.roll(self.sensor_buffer, -1, axis=0)
        
        # Format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp, vibration, altitude, gps_accuracy]
        self.sensor_buffer[-1] = [
            sensor_data["accel"][0], sensor_data["accel"][1], sensor_data["accel"][2],
            sensor_data["gyro"][0], sensor_data["gyro"][1], sensor_data["gyro"][2],
            sensor_data["temperature"],
            sensor_data["vibration"],
            sensor_data["altitude"],
            sensor_data["gps_accuracy"]
        ]
    
    def predict_calibration(self):
        """Predict calibration offsets"""
        if time.time() - self.last_calibration_time < DRONE_CONFIG["calibration_interval"]:
            return None
        
        try:
            # Prepare input data
            input_data = np.expand_dims(self.sensor_buffer, axis=0)
            
            # Predict calibration offsets
            predictions = self.model.predict(input_data)[0]
            
            # Format calibration parameters
            calibration = {
                'gyro_x_offset': float(predictions[0]),
                'gyro_y_offset': float(predictions[1]),
                'gyro_z_offset': float(predictions[2]),
                'accel_x_offset': float(predictions[3]),
                'accel_y_offset': float(predictions[4]),
                'accel_z_offset': float(predictions[5]),
                'timestamp': time.time()
            }
            
            self.calibration_history.append(calibration)
            self.last_calibration_time = time.time()
            return calibration
        except Exception as e:
            logger.error(f"Calibration prediction failed: {e}")
            return None

# =============================================================================
# SWARM COMMUNICATION MODULE
# =============================================================================
class SwarmCommunication:
    def __init__(self, drone_id, role, ip_address):
        self.drone_id = drone_id
        self.role = role
        self.ip_address = ip_address
        self.swarm_members = {}
        self.message_queue = queue.Queue()
        self.udp_socket = None
        self.running = True
        self.listener_thread = threading.Thread(target=self._udp_listener)
        self.sender_thread = threading.Thread(target=self._udp_sender)
        self._initialize_network()
        logger.info(f"Drone {drone_id} communication system initialized")
    
    def _initialize_network(self):
        """Initialize network communication"""
        try:
            # Create UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind(('0.0.0.0', 14550))
            self.udp_socket.settimeout(1.0)
            
            # Start network threads
            self.listener_thread.start()
            self.sender_thread.start()
            logger.info("Network communication started")
        except Exception as e:
            logger.error(f"Network initialization failed: {e}")
    
    def _udp_listener(self):
        """Listen for incoming swarm messages"""
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                message = json.loads(data.decode('utf-8'))
                self._process_message(message, addr[0])
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Message receive error: {e}")
    
    def _udp_sender(self):
        """Send swarm status updates"""
        while self.running:
            try:
                # Broadcast status at regular intervals
                status = self._generate_status_message()
                for member in self.swarm_members.values():
                    if member.ip_address != self.ip_address:  # Don't send to self
                        self.udp_socket.sendto(
                            json.dumps(status).encode('utf-8'),
                            (member.ip_address, 14550)
                        )
                time.sleep(DRONE_CONFIG["swarm_update_interval"])
            except Exception as e:
                logger.error(f"Message send error: {e}")
    
    def _generate_status_message(self):
        """Generate status message for broadcasting"""
        return {
            "drone_id": self.drone_id,
            "role": self.role.name,
            "ip_address": self.ip_address,
            "timestamp": time.time(),
            "position": (0, 0, 0),  # Would be actual position
            "battery": 100.0,  # Would be actual battery
            "status": "OPERATIONAL"
        }
    
    def _process_message(self, message: Dict, source_ip: str):
        """Process incoming swarm message"""
        drone_id = message.get("drone_id")
        if drone_id == self.drone_id:
            return  # Ignore own messages
        
        # Update swarm member information
        self.swarm_members[drone_id] = SwarmMember(
            drone_id=drone_id,
            role=DroneRole[message["role"]],
            state=DroneState(),
            capabilities=[],
            ip_address=source_ip,
            last_seen=time.time()
        )
        logger.debug(f"Received status from drone {drone_id}")
    
    def send_task(self, task: MissionTask, target_id: int):
        """Send a task to a specific drone"""
        if target_id not in self.swarm_members:
            logger.warning(f"Target drone {target_id} not in swarm")
            return False
        
        try:
            message = {
                "type": "TASK_ASSIGNMENT",
                "task": task.__dict__,
                "sender_id": self.drone_id,
                "timestamp": time.time()
            }
            target_ip = self.swarm_members[target_id].ip_address
            self.udp_socket.sendto(
                json.dumps(message).encode('utf-8'),
                (target_ip, 14550)
            )
            return True
        except Exception as e:
            logger.error(f"Task send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Task send failed: {e}")
            return False
    
    def broadcast_alert(self, alert_type: str, location: Tuple[float, float, float]):
        """Broadcast emergency alert to swarm"""
        try:
            message = {
                "type": "EMERGENCY_ALERT",
                "alert": alert_type,
                "location": location,
                "sender_id": self.drone_id,
                "timestamp": time.time()
            }
            for member in self.swarm_members.values():
                if member.ip_address != self.ip_address:
                    self.udp_socket.sendto(
                        json.dumps(message).encode('utf-8'),
                        (member.ip_address, 14550))
            return True
        except Exception as e:
            logger.error(f"Alert broadcast failed: {e}")
            return False
    
    def shutdown(self):
        """Cleanup communication resources"""
        self.running = False
        if self.udp_socket:
            self.udp_socket.close()
        logger.info("Communication system shutdown")

# =============================================================================
# TASK MANAGER MODULE
# =============================================================================
class TaskManager:
    def __init__(self, drone_id, role):
        self.drone_id = drone_id
        self.role = role
        self.task_queue = []
        self.current_task = None
        self.completed_tasks = []
        self.next_task_id = 1
        logger.info(f"Drone {drone_id} task manager initialized")
    
    def create_task(self, task_type, location, priority=5):
        """Create a new mission task"""
        task = MissionTask(
            task_id=self.next_task_id,
            task_type=task_type,
            priority=priority,
            location=location
        )
        self.task_queue.append(task)
        self.next_task_id += 1
        return task
    
    def assign_task(self, task: MissionTask):
        """Assign a task to this drone"""
        if self.current_task is None:
            self.current_task = task
            task.assigned_to = self.drone_id
            task.status = "IN_PROGRESS"
            return True
        return False
    
    def complete_current_task(self, success=True):
        """Mark current task as completed"""
        if self.current_task:
            self.current_task.status = "COMPLETED" if success else "FAILED"
            self.completed_tasks.append(self.current_task)
            self.current_task = None
            return True
        return False
    
    def get_next_task(self):
        """Get the highest priority task"""
        if not self.task_queue:
            return None
        
        # Sort tasks by priority (higher priority first)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        # Return highest priority task
        return self.task_queue[0]
    
    def reassign_tasks(self):
        """Reassign tasks based on current swarm state"""
        # This would implement a sophisticated task allocation algorithm
        # For simplicity, we'll just reassign based on priority
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        # Reassign unassigned tasks
        for task in self.task_queue:
            if task.assigned_to == -1 and task.status == "PENDING":
                # In real system, this would find the best drone for the task
                task.assigned_to = self.drone_id
                task.status = "ASSIGNED"

# =============================================================================
# PATH PLANNING MODULE
# =============================================================================
class PathPlanner:
    def __init__(self):
        self.model = self._load_model(AI_MODEL_PATHS["path_planning"])
        self.current_path = []
        self.obstacles = []
        logger.info("Path planning system initialized")
    
    def _load_model(self, model_path):
        """Load path planning model"""
        # This would be a reinforcement learning model in a real system
        return None
    
    def plan_path(self, start, end, obstacles=None):
        """Plan a path from start to end"""
        # Simplified path planning - in real system use A* or RRT
        path = [start]
        
        # Add intermediate points (simulated)
        if start[0] != end[0] or start[1] != end[1]:
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2
            path.append((mid_lat, mid_lon, start[2]))
        
        path.append(end)
        self.current_path = path
        return path
    
    def avoid_obstacles(self, current_position, obstacles):
        """Adjust path to avoid obstacles"""
        # Simplified obstacle avoidance
        adjusted_path = self.current_path.copy()
        for obstacle in obstacles:
            if self._distance(current_position, obstacle["position"]) < DRONE_CONFIG["safe_distance"]:
                # Create detour point
                detour_point = (
                    current_position[0] + 0.0001,
                    current_position[1] + 0.0001,
                    current_position[2]
                )
                adjusted_path.insert(1, detour_point)
                break
        return adjusted_path
    
    def _distance(self, pos1, pos2):
        """Calculate approximate distance in meters"""
        # Simplified Haversine calculation
        lat1, lon1, _ = pos1
        lat2, lon2, _ = pos2
        R = 6371000  # Earth radius in meters
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

# =============================================================================
# MAIN DRONE AI CONTROLLER
# =============================================================================
class DisasterResponseDrone:
    def __init__(self, drone_id, role, connection_string='/dev/ttyAMA0', baud=57600):
        self.drone_id = drone_id
        self.role = role
        self.state = DroneState()
        self.vehicle = self._connect_to_pixhawk(connection_string, baud)
        self.sensors = SensorInterface(self.vehicle)
        self.perception = AIPerception(drone_id)
        self.communication = SwarmCommunication(drone_id, role, self._get_ip_address())
        self.task_manager = TaskManager(drone_id, role)
        self.calibrator = CalibrationAI()
        self.path_planner = PathPlanner()
        self.running = True
        logger.info(f"Drone {drone_id} AI system initialized")
    
    def _connect_to_pixhawk(self, connection_string, baud):
        """Connect to Pixhawk flight controller"""
        try:
            vehicle = connect(connection_string, wait_ready=True, baud=baud)
            logger.info(f"Connected to flight controller: {vehicle.version}")
            return vehicle
        except Exception as e:
            logger.error(f"Pixhawk connection failed: {e}")
            sys.exit(1)
    
    def _get_ip_address(self):
        """Get the drone's IP address (simplified)"""
        # In real system, this would get the actual IP
        return f"192.168.1.{100 + self.drone_id}"
    
    def run(self):
        """Main execution loop for the drone"""
        try:
            logger.info(f"Starting drone {self.drone_id} mission")
            
            # Pre-flight checks
            self._preflight_checks()
            
            # Takeoff
            self._arm_and_takeoff(DRONE_CONFIG["max_altitude"])
            
            # Main mission loop
            while self.running and self.vehicle.armed:
                try:
                    # Update drone state
                    self._update_state()
                    
                    # Perform AI calibration
                    self._ai_calibration()
                    
                    # Role-specific mission execution
                    if self.role == DroneRole.SCOUT:
                        self._scout_mission_loop()
                    elif self.role == DroneRole.DELIVERY:
                        self._delivery_mission_loop()
                    elif self.role == DroneRole.COMMAND:
                        self._command_mission_loop()
                    
                    # Check battery level
                    if self.state.battery < DRONE_CONFIG["battery_threshold"]:
                        logger.warning("Low battery! Returning to base")
                        self._return_to_base()
                    
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Mission loop error: {e}")
                    self._handle_emergency()
            
            # Land and shutdown
            self._land()
        except KeyboardInterrupt:
            logger.info("Mission interrupted by user")
            self._emergency_land()
        finally:
            self.shutdown()
    
    def _preflight_checks(self):
        """Perform pre-flight checks"""
        logger.info("Starting pre-flight checks")
        
        # Check GPS fix
        while not self.vehicle.gps_0 or self.vehicle.gps_0.fix_type < 3:
            logger.warning("Waiting for GPS fix...")
            time.sleep(1)
        
        # Check battery
        if self.vehicle.battery.level < 50:
            logger.error("Insufficient battery for mission")
            sys.exit(1)
        
        # Check sensors
        if not self.sensors.get_sensor_data(SensorType.IMU):
            logger.error("IMU not functioning")
            sys.exit(1)
        
        logger.info("Pre-flight checks passed")
    
    def _arm_and_takeoff(self, altitude):
        """Arm motors and takeoff to specified altitude"""
        logger.info(f"Arming motors and taking off to {altitude}m")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        
        while not self.vehicle.armed:
            time.sleep(0.5)
        
        self.vehicle.simple_takeoff(altitude)
        
        while abs(self.vehicle.location.global_relative_frame.alt - altitude) > 0.5:
            time.sleep(1)
        
        self.state.status = MissionState.SEARCH
        logger.info("Reached target altitude")
    
    def _update_state(self):
        """Update the drone's state information"""
        self.state.position = (
            self.vehicle.location.global_frame.lat,
            self.vehicle.location.global_frame.lon,
            self.vehicle.location.global_frame.alt
        )
        self.state.velocity = (
            self.vehicle.velocity[0],
            self.vehicle.velocity[1],
            self.vehicle.velocity[2]
        )
        self.state.attitude = (
            self.vehicle.attitude.roll,
            self.vehicle.attitude.pitch,
            self.vehicle.attitude.yaw
        )
        self.state.battery = self.vehicle.battery.level
        self.state.mode = self.vehicle.mode
    
    def _ai_calibration(self):
        """Perform AI-driven sensor calibration"""
        # Collect sensor data for calibration
        sensor_data = {
            "accel": (
                self.vehicle.raw_imu.xacc,
                self.vehicle.raw_imu.yacc,
                self.vehicle.raw_imu.zacc
            ),
            "gyro": (
                self.vehicle.raw_imu.xgyro,
                self.vehicle.raw_imu.ygyro,
                self.vehicle.raw_imu.zgyro
            ),
            "temperature": 25.0,  # Would be actual temp
            "vibration": 0.1,  # Would be actual vibration
            "altitude": self.vehicle.location.global_relative_frame.alt,
            "gps_accuracy": self.vehicle.gps_0.eph
        }
        
        # Update calibration model
        self.calibrator.update_sensor_data(sensor_data)
        
        # Predict and apply calibration
        calibration = self.calibrator.predict_calibration()
        if calibration:
            self.sensors.apply_calibration(calibration)
    
    def _scout_mission_loop(self):
        """Mission execution for scout drones"""
        # Capture and process images
        rgb_frame = self.sensors.capture_rgb_image()
        thermal_frame = self.sensors.capture_thermal_image()
        
        if rgb_frame is not None:
            # Perform object detection
            detections = self.perception.detect_objects(rgb_frame, thermal_frame)
            
            # Process detections
            for detection in detections:
                if detection.class_name == "survivor":
                    # Create rescue task
                    task = self.task_manager.create_task(
                        task_type="RESCUE",
                        location=self._pixel_to_gps(detection.bbox),
                        priority=10
                    )
                    # Assign to delivery drone
                    self._assign_to_delivery_drone(task)
                
                # Handle other detection types
                elif detection.class_name == "fire":
                    self.communication.broadcast_alert("FIRE", self._pixel_to_gps(detection.bbox))
                
                # Update thermal map
                thermal_map = self.perception.generate_thermal_map(thermal_frame)
    
    def _delivery_mission_loop(self):
        """Mission execution for delivery drones"""
        # Check for assigned tasks
        if self.task_manager.current_task is None:
            task = self.task_manager.get_next_task()
            if task:
                self.task_manager.assign_task(task)
        
        # Execute current task
        if self.task_manager.current_task:
            task = self.task_manager.current_task
            
            # Plan path to target
            current_pos = self.state.position
            path = self.path_planner.plan_path(current_pos, task.location)
            
            # Follow path
            for point in path[1:]:  # Skip current position
                self.vehicle.simple_goto(LocationGlobalRelative(*point))
                
                # Check for obstacles
                lidar_dist = self.sensors.get_sensor_data(SensorType.LIDAR)
                obstacles = self.perception.detect_obstacles(lidar_dist)
                
                # Replan if obstacles detected
                if obstacles:
                    path = self.path_planner.avoid_obstacles(self.state.position, obstacles)
                    self.vehicle.simple_goto(LocationGlobalRelative(*path[1]))
                
                # Wait until reached point (simplified)
                time.sleep(2)
            
            # Deliver payload
            if self._distance(self.state.position, task.location) < 5.0:
                self._deliver_payload()
                self.task_manager.complete_current_task()
    
    def _command_mission_loop(self):
        """Mission execution for command drones"""
        # Monitor swarm status
        self._monitor_swarm_health()
        
        # Coordinate tasks
        self._coordinate_swarm()
        
        # Update global map
        self._update_global_map()
    
    def _assign_to_delivery_drone(self, task):
        """Assign task to an appropriate delivery drone"""
        # Find available delivery drones
        delivery_drones = [
            m for m in self.communication.swarm_members.values() 
            if m.role == DroneRole.DELIVERY and m.state.status != MissionState.EMERGENCY
        ]
        
        if delivery_drones:
            # Select nearest drone (simplified)
            selected = min(delivery_drones, key=lambda d: self._distance(
                self.state.position, d.state.position))
            
            # Send task
            if self.communication.send_task(task, selected.drone_id):
                logger.info(f"Task {task.task_id} assigned to drone {selected.drone_id}")
        else:
            logger.warning("No available delivery drones for task")
    
    def _pixel_to_gps(self, bbox):
        """Convert pixel coordinates to GPS position"""
        # Simplified transformation - real system would use camera model
        lat, lon, alt = self.state.position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Arbitrary scaling factors
        lat_offset = (center_x - SENSOR_CONFIG["camera_resolution"][0]/2) * 0.00001
        lon_offset = (center_y - SENSOR_CONFIG["camera_resolution"][1]/2) * 0.00001
        
        return (lat + lat_offset, lon + lon_offset, alt)
    
    def _deliver_payload(self):
        """Deliver payload to current location"""
        logger.info("Delivering payload")
        # Release mechanism (servo control)
        self.vehicle.channels.overrides['9'] = 2000  # Release
        time.sleep(2)
        self.vehicle.channels.overrides['9'] = 1000  # Reset
    
    def _return_to_base(self):
        """Return to launch location"""
        logger.info("Returning to base")
        self.state.status = MissionState.RETURN
        home = self.vehicle.home_location
        if home:
            self.vehicle.simple_goto(home)
            while self._distance(self.state.position, (home.lat, home.lon, home.alt)) > 5.0:
                time.sleep(1)
            self._land()
        else:
            logger.error("Home location not set!")
            self._emergency_land()
    
    def _land(self):
        """Land at current position"""
        logger.info("Landing")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.armed:
            time.sleep(1)
        self.state.status = MissionState.PREFLIGHT
        logger.info("Landed safely")
    
    def _emergency_land(self):
        """Immediate emergency landing"""
        logger.critical("EMERGENCY LANDING!")
        self.vehicle.mode = VehicleMode("LAND")
        self.state.status = MissionState.EMERGENCY
        self.communication.broadcast_alert("EMERGENCY_LANDING", self.state.position)
    
    def _handle_emergency(self):
        """Handle emergency situations"""
        self.state.status = MissionState.EMERGENCY
        self.communication.broadcast_alert("SYSTEM_FAILURE", self.state.position)
        self._emergency_land()
    
    def _distance(self, pos1, pos2):
        """Calculate approximate distance in meters"""
        # Simplified calculation
        return math.sqrt(
            (pos2[0]-pos1[0])**2 + 
            (pos2[1]-pos1[1])**2 + 
            (pos2[2]-pos1[2])**2
        ) * 1e5  # Rough conversion from degrees to meters
    
    def shutdown(self):
        """Cleanup all resources"""
        logger.info("Shutting down drone systems")
        self.running = False
        self.sensors.shutdown()
        self.communication.shutdown()
        self.vehicle.close()
        logger.info("Drone systems shutdown")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disaster Response Drone AI System')
    parser.add_argument('--id', type=int, required=True, help='Drone ID')
    parser.add_argument('--role', choices=['SCOUT', 'DELIVERY', 'COMMAND'], required=True, help='Drone role')
    parser.add_argument('--connection', default='/dev/ttyAMA0', help='Pixhawk connection string')
    parser.add_argument('--baud', type=int, default=57600, help='Connection baud rate')
    args = parser.parse_args()
    
    try:
        role = DroneRole[args.role]
        drone = DisasterResponseDrone(
            drone_id=args.id,
            role=role,
            connection_string=args.connection,
            baud=args.baud
        )
        drone.run()
    except Exception as e:
        logger.critical(f"Critical failure: {e}")
        sys.exit(1)