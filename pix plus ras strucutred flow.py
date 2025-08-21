import cv2
import numpy as np
from pymavlink import mavutil
import threading
import json
import time
from tf_lite_detector import TFLiteDetector  # Assume a helper class

# --- Configuration ---
MASTER_PORT = '/dev/ttyAMA0'
SCOUT_UDP = 'udp:192.168.2.2:14550'
BAUDRATE = 57600

# --- Initialize Components ---
try:
    MASTER = mavutil.mavlink_connection(MASTER_PORT, baud=BAUDRATE)
except Exception as e:
    print(f"Failed to connect to MASTER: {e}")
    exit(1)

try:
    SCOUT_COMPANION = mavutil.mavlink_connection(SCOUT_UDP)
except Exception as e:
    print(f"Failed to connect to SCOUT_COMPANION: {e}")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible.")
    exit(1)

detector = TFLiteDetector('model.tflite', 'labelmap.txt')
poi_list = []
mission_complete = threading.Event()

def wait_for_heartbeat(conn, name=""):
    print(f"Waiting for heartbeat from {name or conn.target_system}...")
    conn.wait_heartbeat()
    print(f"Heartbeat from system (system {conn.target_system} component {conn.target_component})")

wait_for_heartbeat(MASTER, "MASTER")
wait_for_heartbeat(SCOUT_COMPANION, "SCOUT_COMPANION")

def send_mission_item(conn, seq, frame, lat, lon, alt):
    # Send a MISSION_ITEM_INT to the Scout
    try:
        conn.mav.mission_item_int_send(
            conn.target_system,
            conn.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2, 0, 0, 0, 0, 0,
            int(lat * 1e7),
            int(lon * 1e7),
            alt
        )
        print(f"Sent mission item to Scout: seq={seq}, lat={lat}, lon={lon}, alt={alt}")
    except Exception as e:
        print(f"Failed to send mission item: {e}")

def get_gps_location(timeout=5):
    # Listen for GPS_RAW_INT messages and return lat, lon, alt
    start = time.time()
    while time.time() - start < timeout:
        msg = MASTER.recv_match(type='GPS_RAW_INT', blocking=True, timeout=1)
        if msg:
            return (msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1e3)
    print("GPS data not available.")
    return (0.0, 0.0, 0.0)

def check_mission_complete():
    # Dummy logic: stop after 100 POIs or 'q' pressed
    return len(poi_list) >= 100

def main_mission():
    seq = 1
    while not mission_complete.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Run Object Detection
        boxes, classes, scores = detector.detect(frame)
        current_location = get_gps_location()

        for i in range(len(classes)):
            if scores[i] > 0.7:
                class_name = detector.get_class_name(classes[i])
                print(f"Detected: {class_name} at {current_location}")

                # Log the POI
                poi = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [current_location[1], current_location[0]]
                    },
                    "properties": {
                        "object_class": class_name,
                        "confidence": float(scores[i])
                    }
                }
                poi_list.append(poi)

                # If it's a class for the Scout (e.g., 'person', 'gas_leak')
                if class_name in ['person', 'gas_leak']:
                    send_mission_item(SCOUT_COMPANION, seq, 10, current_location[0], current_location[1], current_location[2])
                    seq += 1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mission_complete.set()
            break

        if check_mission_complete():
            print("Mission complete condition met.")
            mission_complete.set()
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate Report
    feature_collection = {"type": "FeatureCollection", "features": poi_list}
    with open('mission_report.geojson', 'w') as f:
        json.dump(feature_collection, f, indent=2)
    print("Report generated: mission_report.geojson")

if __name__ == "__main__":
    try:
        main_mission()
    except KeyboardInterrupt:
        print("Mission interrupted by user.")
        mission_complete.set()
        cap.release()
        cv2.destroyAllWindows()
        feature_collection = {"type": "FeatureCollection", "features": poi_list}
        with open('mission_report.geojson', 'w') as f:
            json.dump(feature_collection, f, indent=2)