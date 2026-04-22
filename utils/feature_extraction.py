import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sqlalchemy import Extract

class FeatureExtractor:
    def __init__(self):
        # Fitur: Average RGB (3) + GLCM (4) = 7 features total
        self.feature_names = ['avg_red', 'avg_green', 'avg_blue', 
                             'contrast', 'homogeneity', 'correlation', 'energy']
    
    def extract_rgb_average(self, image_rgb):
        """Extract average values for R, G, B channels
        
        Parameters:
        - image_rgb: RGB image (uint8)
        
        Return: list [avg_red, avg_green, avg_blue]
        """
        if image_rgb is None:
            print("[WARNING] image_rgb is None in extract_rgb_average")
            return [0.0, 0.0, 0.0]
        
        try:
            avg_red = np.mean(image_rgb[:, :, 0])
            avg_green = np.mean(image_rgb[:, :, 1])
            avg_blue = np.mean(image_rgb[:, :, 2])
            return [avg_red, avg_green, avg_blue]
        except Exception as e:
            print(f"[ERROR] extract_rgb_average failed: {str(e)}")
            return [0.0, 0.0, 0.0]
    
    def glcm_features(self, gray_image):
        """Extract GLCM texture features (4)
        
        Parameters:
        - gray_image: Grayscale image hasil threshold (uint8)
        
        Return: list [contrast, homogeneity, correlation, energy]
        """
        # Validate input
        if gray_image is None:
            print("[WARNING] gray_image is None in glcm_features")
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            gray = gray_image.astype(np.uint8)
            
            glcm = graycomatrix(gray, distances=[1, 2, 3],
                               angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True, normed=True)
            
            features = []
            for prop in ['contrast', 'homogeneity', 'correlation', 'energy']:
                prop_values = graycoprops(glcm, prop)
                features.append(np.mean(prop_values))
            
            return features
        except Exception as e:
            print(f"[ERROR] glcm_features failed: {str(e)}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def extract_all_features(self, image_rgb, gray_processed):
        """Extract 7 features: Average RGB (3) + GLCM (4)
        
        Parameters:
        - image_rgb: RGB image from preprocessing pipeline (uint8)
        - gray_processed: Grayscale hasil threshold dari preprocessing pipeline (uint8)
        
        Return: list of 7 features
        """
        # Validate inputs
        if image_rgb is None:
            print("[ERROR] image_rgb is None in extract_all_features")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if gray_processed is None:
            print("[ERROR] gray_processed is None in extract_all_features")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Extract average RGB values (3)
        rgb_features = self.extract_rgb_average(image_rgb)
        
        # Extract GLCM texture features (4)
        glcm_features_list = self.glcm_features(gray_processed)
        
        all_features = rgb_features + glcm_features_list
        
        return all_features
    
    def save_features_to_csv(self, features_list, labels, filename):
        """Save extracted features to CSV"""
        df = pd.DataFrame(features_list, columns=self.feature_names)
        df['label'] = labels
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Fitur disimpan ke: {filename}")
        
        return df