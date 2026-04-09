"""Video augmentation transforms using OpenCV (CPU only)."""

import cv2
import numpy as np


def color_jitter(frame: np.ndarray, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1) -> np.ndarray:
    """Apply random color jitter to a BGR frame.

    Each parameter controls the max random deviation from the original.
    """
    # Work in float32 to avoid clipping between operations
    result = frame.astype(np.float32)

    # Brightness: shift pixel values up/down by a random amount
    b_delta = np.random.uniform(-brightness, brightness) * 255
    result += b_delta

    # Contrast: scale deviation from the mean intensity
    c_factor = np.random.uniform(1 - contrast, 1 + contrast)
    mean = result.mean()
    result = (result - mean) * c_factor + mean

    # Switch to HSV to manipulate saturation and hue independently
    hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation: scale the S channel
    s_factor = np.random.uniform(1 - saturation, 1 + saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)

    # Hue: rotate the H channel (OpenCV range 0-180, not 0-360)
    h_delta = np.random.uniform(-hue, hue) * 180
    hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to a frame."""
    # OpenCV requires an odd kernel size; round up if even
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def gaussian_noise(frame: np.ndarray, mean: float = 0.0, std: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to a frame."""
    noise = np.random.normal(mean, std, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def sharpen(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Sharpen a frame using an unsharp mask technique."""
    # Unsharp mask: subtract a blurred copy to amplify edges
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    # result = original * (1 + strength) - blurred * strength
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return sharpened


TRANSFORMS = {
    "color_jitter": color_jitter,
    "blur": blur,
    "gaussian_noise": gaussian_noise,
    "sharpen": sharpen,
}


def apply_transforms(frame: np.ndarray, transform_names: list[str], **kwargs) -> np.ndarray:
    """Apply a sequence of named transforms to a frame.

    Kwargs are shared across all transforms; each function only receives
    the kwargs matching its signature (e.g. blur gets kernel_size, not hue).
    """
    for name in transform_names:
        fn = TRANSFORMS[name]
        import inspect
        sig = inspect.signature(fn)
        valid_keys = [p for p in sig.parameters if p != "frame"]
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        frame = fn(frame, **filtered)
    return frame
