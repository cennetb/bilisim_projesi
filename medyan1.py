import cv2
def apply_median_filter(image, kernel_size):

    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image




