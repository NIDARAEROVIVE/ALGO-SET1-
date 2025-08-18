def geotag_detection(frame, gps_data):
    # Embed metadata
    exif_data = {
        'GPSLatitude': gps_data.lat,
        'GPSLongitude': gps_data.lon,
        'GPSAltitude': gps_data.alt,
        'DateTimeOriginal': datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    }
    
    # Save image with metadata
    cv2.imwrite('detection.jpg', frame)
    with open('detection.jpg', 'a+') as img_file:
        piexif.insert(piexif.dump({'GPS': exif_data}), img_file)
    
    # Create GIS marker
    return GeoJSONFeature(
        geometry=Point((gps_data.lon, gps_data.lat)),
        properties={'urgency': calculate_urgency(frame)}
    )