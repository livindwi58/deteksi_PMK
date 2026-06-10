import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import pandas as pd


class FeatureExtractor:
    def __init__(self, n_hist_bins=8):
        self.n_hist_bins = n_hist_bins
        self.feature_names = [
            # RGB average (3)
            'avg_red', 'avg_green', 'avg_blue',
            # HSV mean (3)
            'mean_hue', 'mean_saturation', 'mean_value',
            # HSV std (3)
            'std_hue', 'std_saturation', 'std_value',
            # GLCM (6)
            'contrast', 'homogeneity', 'correlation', 'energy',
            'dissimilarity', 'ASM',
            # Color histogram (3 channels x n_hist_bins)
            *[f'hist_r_{i}' for i in range(n_hist_bins)],
            *[f'hist_g_{i}' for i in range(n_hist_bins)],
            *[f'hist_b_{i}' for i in range(n_hist_bins)],
            # Hu moments (7)
            *[f'hu_moment_{i+1}' for i in range(7)],
        ]

    def extract_rgb_average(self, image_rgb):
        if image_rgb is None:
            return [0.0, 0.0, 0.0]
        try:
            return [np.mean(image_rgb[:, :, c]) for c in range(3)]
        except Exception:
            return [0.0, 0.0, 0.0]

    def extract_hsv_features(self, image_rgb):
        if image_rgb is None:
            return [0.0] * 6
        try:
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] *= 360.0 / 180.0  # hue 0-360
            means = [np.mean(hsv[:, :, c]) for c in range(3)]
            stds = [np.std(hsv[:, :, c]) for c in range(3)]
            return means + stds
        except Exception:
            return [0.0] * 6

    def glcm_features(self, gray_image):
        if gray_image is None:
            return [0.0] * 6
        try:
            gray = gray_image.astype(np.uint8)
            glcm = graycomatrix(gray, distances=[1, 2, 3],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=256, symmetric=True, normed=True)
            props = ['contrast', 'homogeneity', 'correlation', 'energy',
                     'dissimilarity', 'ASM']
            return [np.mean(graycoprops(glcm, prop)) for prop in props]
        except Exception:
            return [0.0] * 6

    def extract_color_histogram(self, image_rgb):
        if image_rgb is None:
            return [0.0] * (3 * self.n_hist_bins)
        try:
            features = []
            for c in range(3):
                hist = cv2.calcHist([image_rgb], [c], None,
                                    [self.n_hist_bins], [0, 256])
                hist = hist.flatten() / (image_rgb.shape[0] * image_rgb.shape[1])
                features.extend(hist.tolist())
            return features
        except Exception:
            return [0.0] * (3 * self.n_hist_bins)

    def extract_hu_moments(self, gray_processed):
        if gray_processed is None:
            return [0.0] * 7
        try:
            binary = (gray_processed > 0).astype(np.uint8) * 255
            moments = cv2.moments(binary)
            hu = cv2.HuMoments(moments)
            # Log-scale Hu moments for numerical stability
            hu = [-np.sign(h) * np.log10(np.abs(h) + 1e-10) for h in hu.flatten()]
            return hu
        except Exception:
            return [0.0] * 7

    def extract_all_features(self, image_rgb, gray_processed):
        if image_rgb is None:
            print("[ERROR] image_rgb is None in extract_all_features")
            return [0.0] * len(self.feature_names)
        if gray_processed is None:
            print("[ERROR] gray_processed is None in extract_all_features")
            return [0.0] * len(self.feature_names)

        features = []
        features.extend(self.extract_rgb_average(image_rgb))
        features.extend(self.extract_hsv_features(image_rgb))
        features.extend(self.glcm_features(gray_processed))
        features.extend(self.extract_color_histogram(image_rgb))
        features.extend(self.extract_hu_moments(gray_processed))
        return features

    def save_features_to_csv(self, features_list, labels, filename):
        df = pd.DataFrame(features_list, columns=self.feature_names)
        df['label'] = labels
        df.to_csv(filename, index=False)
        print(f"Fitur disimpan ke: {filename}")
        return df