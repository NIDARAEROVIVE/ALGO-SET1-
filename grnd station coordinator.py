while mission_active:
    # Dynamic area partitioning
    sectors = voronoi_partition(flood_area, [scout_pos, delivery_pos])
    
    # Task assignment
    if new_survivor:
        if survivor['urgency'] > URGENCY_THRESHOLD:
            assign_delivery(scout_last_known_pos)
        else:
            add_to_delivery_queue()
    
    # Scout directives
    send_command(scout, {
        'type': 'scan_sector',
        'polygon': sectors['scout'],
        'altitude': adjust_alt(terrain_roughness)
    })
    
    # Delivery management
    if delivery.state == 'IDLE' and not delivery_queue.empty():
        next_target = delivery_queue.pop()
        send_command(delivery, {
            'type': 'deliver_payload',
            'target': next_target.coords,
            'payload_type': select_payload(next_target.urgency)
        })
    
    # Live map update
    update_map(scout_telemetry, delivery_telemetry, survivor_data)