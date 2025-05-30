import numpy as np
import cv2


class ColorTransfer:

    def rgb2lab(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    def lab2rgb(self, lab: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def rgb2yuv(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)

    def yuv2rgb(self, yuv: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    def srgb2lin(self, s):
        s = s.astype(float) / 255.0
        return np.where(
            s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
        )

    def lin2srgb(self, lin):
        return 255 * np.where(
            lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
        )

    def get_luminance(self, linear_image: np.ndarray):
        return np.sum(linear_image * [0.2126, 0.7152, 0.0722], axis=2)

    def take_luminance_from_first_chroma_from_second(
        self, luminance, chroma, mode="lab", s=1
    ):
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
