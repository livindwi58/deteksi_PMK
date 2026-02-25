import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'mean_r', 'mean_g', 'mean_b',
            'std_r', 'std_g', 'std_b',
            'skew_r', 'skew_g', 'skew_b',
            'energy', 'contrast', 'correlation',
            'homogeneity', 'dissimilarity', 'ASM'
        ]
    
    def color_moments(self, image):
        """Extract color moments from RGB image

        Supports masked ROI when `mask` is provided inside `image` tuple
        (see `extract_all_features`). If `image` is a numpy array, no mask
        is used.
        """
        moments = []

        # If caller passed a tuple (image, mask) handle it; otherwise mask=None
        if isinstance(image, tuple) and len(image) == 2:
            img, mask = image
        else:
            img, mask = image, None

        for i in range(3):  # For each channel (R, G, B)
            channel = img[:, :, i]

            if mask is not None:
                mask_bool = (mask > 0)
                vals = channel[mask_bool]
            else:
                vals = channel.ravel()

            if vals.size == 0:
                mean = 0.0
                std = 0.0
                skew = 0.0
            else:
                mean = np.mean(vals)
                std = np.std(vals)
                if std > 0:
                    skew = np.mean((vals - mean) ** 3) / (std ** 3)
                else:
                    skew = 0.0

            moments.extend([mean, std, skew])

        return moments
    
    def glcm_features(self, image):
        """Extract GLCM texture features.

        Accepts either `image` (numpy array) or `(image, mask)` tuple.
        If `mask` is provided, compute GLCM on cropped ROI to avoid
        background influence.
        """
        # Unpack possible (image, mask)
        if isinstance(image, tuple) and len(image) == 2:
            img, mask = image
        else:
            img, mask = image, None

        # Convert to grayscale and scale to 0-255
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # If mask provided, crop to bounding box of mask
        if mask is not None:
            mask_bool = (mask > 0)
            if mask_bool.any():
                ys, xs = np.where(mask_bool)
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                gray_crop = gray[ymin:ymax+1, xmin:xmax+1]
                # Use crop when it contains enough pixels
                if gray_crop.size >= 16:
                    gray = gray_crop

        gray = gray.astype(np.uint8)

        # Calculate GLCM
        glcm = graycomatrix(gray, 
                           distances=[1, 2, 3],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256,
                           symmetric=True,
                           normed=True)

        # Extract properties and average across distances and angles
        features = []
        for prop in ['energy', 'contrast', 'correlation', 'homogeneity', 'dissimilarity', 'ASM']:
            prop_values = graycoprops(glcm, prop)
            features.append(np.mean(prop_values))

        return features
    
    def extract_all_features(self, image, mask=None):
        """Extract all features: color moments + GLCM.

        `image` is expected to be an RGB array normalized to 0..1. If a
        `mask` is provided it will be used by subroutines; alternatively
        a tuple `(image, mask)` may be passed in `image`.
        """
        # Allow either passing mask separately or embedding into image tuple
        if mask is not None:
            img_for_color = (image, mask)
            img_for_glcm = (image, mask)
        else:
            img_for_color = image
            img_for_glcm = image

        color_features = self.color_moments(img_for_color)
        texture_features = self.glcm_features(img_for_glcm)

        all_features = color_features + texture_features
        return all_features
    
    def save_features_to_csv(self, features_list, labels, filename):
        """Save extracted features to CSV"""
        df = pd.DataFrame(features_list, columns=self.feature_names)
        df['label'] = labels
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Fitur disimpan ke: {filename}")
        
        return df