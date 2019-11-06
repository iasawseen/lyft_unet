

def get_class_stats():
    class_to_size = {
        "car": (1.93, 4.76, 1.72),
        "motorcycle": (0.96, 2.35, 1.59),
        "bus": (2.96, 12.34, 3.44),
        "bicycle": (0.63, 1.76, 1.44),
        "truck": (2.84, 10.24, 3.44),
        "pedestrian": (0.77, 0.81, 1.78),
        "other_vehicle": (2.79, 8.20, 3.23),
        "animal": (0.36, 0.73, 0.51),
        "emergency_vehicle": (2.45, 6.52, 2.39)
    }

    class_to_width = {class_name: class_to_size[class_name][0] for class_name in class_to_size}
    class_to_len = {class_name: class_to_size[class_name][1] for class_name in class_to_size}
    class_to_height = {class_name: class_to_size[class_name][2] for class_name in class_to_size}

    return class_to_width, class_to_len, class_to_height


def get_classes():
    return [
        "car", "motorcycle", "bus", "bicycle", "truck",
        "pedestrian", "other_vehicle", "animal", "emergency_vehicle"
    ]
