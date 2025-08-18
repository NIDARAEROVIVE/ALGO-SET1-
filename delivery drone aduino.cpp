void loop() {
  checkMavlinkMessages();
  
  switch(mission_state) {
    case STANDBY:
      if (new_delivery_order) {
        calculate_path(target);
        mission_state = NAVIGATING;
      }
      break;
      
    case NAVIGATING:
      follow_path();
      if (distance_to(target) < 5.0) {
        mission_state = DEPLOYING;
      }
      break;
      
    case DEPLOYING:
      altitude_hold(15.0);  // Maintain 15m height
      release_payload(selected_payload);
      if (payload_released) {
        send_confirmation();
        mission_state = RETURNING;
      }
      break;
      
    case RETURNING:
      navigate_home();
      if (at_home()) {
        mission_state = STANDBY;
      }
      break;
  }
  
  // Emergency override
  if (received_emergency_drop) {
    abort_current_mission();
    target = emergency_location;
    mission_state = NAVIGATING;
  }
  
  obstacle_avoidance_check();
}