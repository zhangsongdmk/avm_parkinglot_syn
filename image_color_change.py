import cv2


def change_slot_color(image):


    brightness = -80
    bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)

    return bright_image
    