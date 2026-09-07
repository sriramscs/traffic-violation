"""
Traffic Violation Detection Rules & Spatial Association Heuristics.
Handles multi-violation verification:
- Triple Riding (Motorcycle + Rider matching)
- Helmet / No Helmet region extraction (Head ROI)
- Seatbelt / No Seatbelt ROI extraction (Driver chest/windshield ROI)
- Mobile Phone usage proximity
"""

import cv2
import numpy as np


def is_rider_on_motorcycle(person_box, bike_box, x_tolerance=25):
    """
    Checks if a detected person is mounted on a motorcycle.
    Accounts for rider torso and head extending above bike bounding box.
    """
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    p_center_x = (px1 + px2) // 2
    p_bottom_y = py2

    # Expanded bike upper zone to encapsulate mounted riders
    bike_expanded_top = by1 - int((by2 - by1) * 0.85)

    horizontal_match = (bx1 - x_tolerance) <= p_center_x <= (bx2 + x_tolerance)
    vertical_match = bike_expanded_top <= p_bottom_y <= (by2 + 30)

    # Intersection over Rider Area (rider should overlap vertically with the bike)
    overlap_y1 = max(py1, by1)
    overlap_y2 = min(py2, by2)
    has_vertical_overlap = (overlap_y2 - overlap_y1) > 0 if overlap_y2 > overlap_y1 else False

    return horizontal_match and (vertical_match or has_vertical_overlap)


def extract_head_crop(frame, person_box, margin_ratio=0.10):
    """
    Extracts the head/helmet region from a detected person's bounding box.
    Typically corresponds to the top 25-30% of the person height.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None, (0, 0, 0, 0)

    # Top 28% of person bounding box
    head_y1 = max(0, int(y1 - box_h * margin_ratio))
    head_y2 = min(h, int(y1 + box_h * 0.28))
    head_x1 = max(0, int(x1 - box_w * margin_ratio))
    head_x2 = min(w, int(x2 + box_w * margin_ratio))

    crop = frame[head_y1:head_y2, head_x1:head_x2]
    return crop, (head_x1, head_y1, head_x2, head_y2)


def extract_driver_chest_crop(frame, car_box):
    """
    Extracts the estimated driver windshield / chest region from a car box
    to analyze seatbelt presence.
    Typically front-left or front-center of vehicle.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = car_box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None, (0, 0, 0, 0)

    # Windshield/driver area is typically upper-middle region of the car
    cx1 = max(0, int(x1 + box_w * 0.15))
    cx2 = min(w, int(x1 + box_w * 0.85))
    cy1 = max(0, int(y1 + box_h * 0.15))
    cy2 = min(h, int(y1 + box_h * 0.65))

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def check_phone_near_person(phone_box, person_box, max_dist_ratio=0.35):
    """
    Determines if a detected mobile phone is in close proximity
    to a person's upper body / head region (active phone usage).
    """
    px1, py1, px2, py2 = person_box
    phx1, phy1, phx2, phy2 = phone_box

    p_w = px2 - px1
    p_h = py2 - py1

    phone_cx = (phx1 + phx2) / 2.0
    phone_cy = (phy1 + phy2) / 2.0

    # Person upper body reference point (near ears/head/chest)
    upper_cx = (px1 + px2) / 2.0
    upper_cy = py1 + p_h * 0.25

    dist = np.sqrt((phone_cx - upper_cx) ** 2 + (phone_cy - upper_cy) ** 2)
    max_allowed_dist = max(p_w, p_h) * max_dist_ratio

    # Also check bounding box overlap
    intersects = not (phx2 < px1 or phx1 > px2 or phy2 < py1 or phy1 > (py1 + p_h * 0.65))

    return (dist <= max_allowed_dist) or intersects
