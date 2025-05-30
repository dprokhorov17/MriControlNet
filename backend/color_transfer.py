"""Color transfer utilities for image processing.

This module provides functionality to transfer colors between images using different
color spaces (LAB, YUV) and methods (luminance-based, color space-based).
"""
import numpy as np
import cv2


class ColorTransfer:
    """Handles color transfer between source and target images.

    Supports multiple color transfer methods:
    - LAB color space transfer
    - YUV color space transfer
    - Luminance-based transfer
    """

    def rgb2lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB image to LAB color space."""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    def lab2rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB image back to RGB color space."""
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def rgb2yuv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB image to YUV color space."""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)

    def yuv2rgb(self, yuv: np.ndarray) -> np.ndarray:
        """Convert YUV image back to RGB color space."""
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    def srgb2lin(self, s):
        """Convert sRGB values to linear RGB space."""
        s = s.astype(float) / 255.0
        return np.where(
            s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
        )

    def lin2srgb(self, lin):
        """Convert linear RGB values back to sRGB space."""
        return 255 * np.where(
            lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
        )

    def get_luminance(self, linear_image: np.ndarray):
        """Calculate luminance from linear RGB image using standard coefficients."""
        return np.sum(linear_image * [0.2126, 0.7152, 0.0722], axis=2)

    def take_luminance_from_first_chroma_from_second(
        self, luminance, chroma, mode="lab", s=1
    ):
        """Transfer luminance from one image while preserving chromaticity of another.

        Args:
            luminance (np.ndarray): Source image for luminance
            chroma (np.ndarray): Source image for color information
            mode (str): Color transfer mode ('lab', 'yuv', or 'luminance')
            s (float): Strength of the transfer (0-1)

        Returns:
            np.ndarray: Resulting image with combined luminance and chromaticity
        """
        assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"

        if mode == "lab":
            lab = self.rgb2lab(chroma)
            lab[:, :, 0] = self.rgb2lab(luminance)[:, :, 0]
            return self.lab2rgb(lab)
        elif mode == "yuv":
            yuv = self.rgb2yuv(chroma)
            yuv[:, :, 0] = self.rgb2yuv(luminance)[:, :, 0]
            return self.yuv2rgb(yuv)
        elif mode == "luminance":
            lluminance = self.srgb2lin(luminance)
            lchroma = self.srgb2lin(chroma)
            return self.lin2srgb(
                np.clip(
                    lchroma
                    * (
                        (self.get_luminance(lluminance) / (self.get_luminance(lchroma)))
                        ** s
                    )[:, :, np.newaxis],
                    0,
                    1,
                )
            )
        else:
            raise ValueError(f"Unsupported color transfer mode: {mode}")
