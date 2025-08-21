#include <SoftwareSerial.h>
#include <mavlink.h> // https://github.com/mavlink/c_library_v2

SoftwareSerial mavlinkSerial(10, 11); // RX, TX for Pixhawk TELEM2

// Sensor Pins
const int trigPin = 5;
const int echoPin = 6;
const int gasSensorPin = A0;

void setup() {
  Serial.begin(57600); // USB for debugging
  mavlinkSerial.begin(57600); // MAVLink telemetry to Pixhawk

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  // 1. CHECK FOR INCOMING MAVLINK MESSAGES
  mavlink_message_t msg;
  mavlink_status_t status;
  
  while(mavlinkSerial.available() > 0) {
    uint8_t c = mavlinkSerial.read();
    if(mavlink_parse_char(MAVLINK_COMM_0, c, &msg, &status)) {
      handleMessage(&msg);
    }
  }

  // 2. IF WE ARE IN CLOSE-INSPECTION MODE, RUN SENSOR LOGIC
  if (inspectionMode) {
    long duration, distance;
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = (duration / 2) / 29.1; // Convert to cm

    int gasValue = analogRead(gasSensorPin);

    // If too close to an obstacle, send a velocity command to move back
    if (distance < 50) { // 50cm threshold
      send_mavlink_velocity_command(0, 0, -1.0); // Move up at 1 m/s
    }

    // Package and send sensor data
    send_mavlink_debug_float("GAS", gasValue);
    send_mavlink_debug_float("DIST", distance);
  }
  delay(100);
}

void handleMessage(mavlink_message_t* msg) {
  switch(msg->msgid) {
    case MAVLINK_MSG_ID_MISSION_ITEM_INT:
      {
        mavlink_mission_item_int_t mission_item;
        mavlink_msg_mission_item_int_decode(msg, &mission_item);
        
        target_lat = mission_item.x;
        target_lon = mission_item.y;
        target_alt = mission_item.z;
        
        // Command Pixhawk to go to waypoint (simplified)
        // This is complex and requires setting the flight mode and uploading a mission item.
        // Often easier to use a GCS to handle this via the Pixhawk's built-in logic.
        inspectionMode = false; // We are not there yet
        break;
      }
  }
}

// Stub function to send velocity commands
void send_mavlink_velocity_command(float vx, float vy, float vz) {
  mavlink_message_t msg;
  uint8_t buf[MAVLINK_MAX_PACKET_LEN];
  
  // Set position target (type mask indicates use velocity)
  mavlink_msg_set_position_target_local_ned_pack(0xFF, 0xBE, &msg, 0,
    MASTER_SYSTEM_ID, 1,
    MAV_FRAME_LOCAL_NED,
    0b0000111111000111, // Type Mask: Ignore pos/accel, use vel/yaw
    0, 0, 0, // x, y, z pos (ignored)
    vx, vy, vz, // vx, vy, vz
    0, 0, 0, // afx, afy, afz (ignored)
    0, 0); // yaw, yaw_rate
  
  uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
  mavlinkSerial.write(buf, len);
}