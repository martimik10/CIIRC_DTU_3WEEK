import numpy as np
import cv2


def colorizeDepthFrame(
    depth_frame: np.ndarray, camera_belt_dist_mm: int = 785
) -> np.ndarray:
    """
    Colorizes provided one channel depth frame into RGB image.

    Args:
        depth_frame (np.ndarray): Depth frame.
        camera_belt_dist_mm (int): Distance from camera to conveyor belt in millimeters.

    Returns:
        np.ndarray: Colorized depth frame.
    """

    # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 5))
    # depth_frame_hist = clahe.apply(depth_frame.astype(np.uint8))
    # colorized_depth_frame = cv2.applyColorMap(depth_frame_hist, cv2.COLORMAP_JET)
    depth_frame = np.clip(
        depth_frame, camera_belt_dist_mm + 30 - 255, camera_belt_dist_mm + 30
    )
    depth_frame -= camera_belt_dist_mm + 30 - 255
    depth_frame = depth_frame.astype(np.uint8)
    max_val = np.max(depth_frame)
    min_val = np.min(depth_frame)
    depth_frame = (depth_frame - min_val) * (255 / (max_val - min_val)) * (-1) + 255
    depth_frame = depth_frame.astype(np.uint8)
    colored = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

    return colored


def drawText(
    frame: np.ndarray, text: str, position: tuple[int, int], size: float = 1
) -> None:
    """
    Draws white text with black border to the frame.

    Args:
        frame (np.ndarray): Frame into which the text will be draw.
        text (str): Text to draw.
        position (tuple[int, int]): Position on the frame in pixels.
        size (float): Size modifier of the text.
    """

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4)
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2
    )


def show_boot_screen(message: str, resolution: tuple[int, int] = (540, 960)):
    """
    Opens main frame window with boot screen message.

    Args:
        message (str): Message to be displayed.
        resolution (tuple[int, int]): Resolution of the window.
    """

    boot_screen = np.zeros(resolution)
    cv2.namedWindow("Frame")
    cv2.putText(
        boot_screen,
        message,
        ((resolution[1] // 2) - 150, resolution[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.imshow("Frame", boot_screen)
    cv2.waitKey(1)
