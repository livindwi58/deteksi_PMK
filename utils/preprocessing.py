import cv2
import numpy as np
from PIL import Image


def validate_cattle_image(image_path, confidence_threshold=0.75):
    """
    Validasi apakah gambar adalah sapi atau bukan.
    Bisa deteksi wajah sapi (dengan mata) atau bagian tubuh sapi lainnya.
    
    Returns: (is_cattle, confidence, reason)
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_img = Image.open(image_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return False, 0.0, "Format gambar tidak didukung"
        
        if img is None:
            return False, 0.0, "Gagal membaca gambar"
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ===== LAYER 1: DETEKSI MATA (OPTIONAL - hanya bonus jika ada) =====
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=30,
            param1=70, param2=35,
            minRadius=5, maxRadius=70
        )
        
        valid_eyes = []
        eye_score = 0.0
        
        if circles is not None and len(circles[0]) > 0:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                if roi.size > 0:
                    roi_mean = np.mean(roi)
                    roi_std = np.std(roi)
                    if roi_mean < 105 and roi_std > 12:
                        valid_eyes.append((x, y, r))
        
        # Validasi mata jika terdeteksi (untuk bonus score)
        if len(valid_eyes) >= 2:
            eye_positions = [(x, y) for x, y, r in valid_eyes[:2]]
            eye_x = sorted([x for x, y in eye_positions])
            eye_y = [valid_eyes[i][1] for i in range(2)]
            eye_distances = [abs(eye_x[1] - eye_x[0]), abs(eye_y[0] - eye_y[1])]
            
            # Mata harus horizontal (left-right)
            if eye_distances[0] >= 20 and eye_distances[1] <= eye_distances[0] * 0.3:
                eye_score = 1.0
                print(f"[VALIDATE] Mata terdeteksi | Spacing: h={eye_distances[0]:.0f}, v={eye_distances[1]:.0f}")
        
        
        # ===== LAYER 2: VALIDASI WARNA SAPI (FLEKSIBEL) =====
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = img_hsv[:, :, 0].astype(np.float32)
        s_channel = img_hsv[:, :, 1].astype(np.float32)
        v_channel = img_hsv[:, :, 2].astype(np.float32)
        
        # Warna sapi: coklat, merah, hitam, putih (bisa partial)
        brown_mask = ((h_channel >= 8) & (h_channel <= 30) & (s_channel >= 30) & (v_channel >= 60))
        red_mask = (((h_channel <= 3) | (h_channel >= 175)) & (s_channel >= 35) & (v_channel >= 50))
        black_mask = ((v_channel < 90) & (s_channel >= 15))
        white_mask = ((v_channel >= 210) & (s_channel <= 50))
        
        cattle_color = brown_mask | red_mask | black_mask | white_mask
        color_ratio = np.sum(cattle_color) / (h * w)
        
        # Minimum warna sapi harus 30% (ketat untuk menolak bukan sapi)
        if color_ratio < 0.30:
            return False, 0.0, "Gambar tidak menunjukkan sapi"
        
        # Score color: ketat untuk matching sapi
        if color_ratio >= 0.50:
            color_score = 1.0
        elif color_ratio >= 0.40:
            color_score = 0.95
        elif color_ratio >= 0.30:
            color_score = 0.80
        else:
            color_score = 0.60
        print(f"[VALIDATE] Warna sapi: {color_ratio*100:.0f}%")
        
        # ===== LAYER 3: VALIDASI TEKSTUR KULIT =====
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Texture sapi harus dalam range spesifik 80-2200 (ketat)
        if texture_variance < 80 or texture_variance > 2200:
            return False, 0.0, "Gambar tidak menunjukkan tekstur sapi"
        
        # Score texture: ketat untuk matching sapi texture
        if 120 <= texture_variance <= 1800:
            texture_score = 1.0
        elif 90 <= texture_variance <= 2100:
            texture_score = 0.90
        else:
            texture_score = 0.75
        print(f"[VALIDATE] Tekstur variance: {texture_variance:.0f}")
        
        # ===== LAYER 4: EDGE DETECTION (Struktur) =====
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Edge density: 2-25% ketat untuk sapi
        if edge_density < 0.02 or edge_density > 0.25:
            return False, 0.0, "Gambar tidak memiliki struktur sapi"
        
        # Score edge: ketat untuk matching
        if 0.05 <= edge_density <= 0.15:
            edge_score = 1.0
        elif 0.03 <= edge_density <= 0.22:
            edge_score = 0.90
        else:
            edge_score = 0.70
        print(f"[VALIDATE] Edge density: {edge_density*100:.1f}%")
        
        # ===== HARD REQUIREMENTS =====
        # REQUIREMENT 1: Warna harus minimal decent (tidak boleh terlalu rendah)
        if color_score < 0.75:
            return False, 0.0, "Warna gambar tidak sesuai sapi"
        
        # REQUIREMENT 2: Texture harus dalam range sapi
        if texture_score < 0.70:
            return False, 0.0, "Tekstur gambar tidak sesuai sapi"
        
        # REQUIREMENT 3: Edge harus terdeteksi
        if edge_score < 0.65:
            return False, 0.0, "Struktur gambar tidak terlihat jelas"
        
        # ===== FINAL CALCULATION =====
        # Ketat: semua feature harus bagus untuk lolos
        final_confidence = (
            eye_score * 0.15 +         # Mata (15% - bonus)
            color_score * 0.40 +       # Warna (40% - paling penting)
            texture_score * 0.30 +     # Tekstur (30% - penting)
            edge_score * 0.15          # Edge (15% - supporting)
        )
        
        print(f"[VALIDATE] Final Score: {final_confidence*100:.1f}% (threshold: {confidence_threshold*100:.0f}%)")
        
        if final_confidence >= confidence_threshold:
            return True, final_confidence, None
        else:
            return False, final_confidence, "Bukan sapi atau gambar kurang jelas"
    
    except Exception as e:
        print(f"[VALIDATE] ERROR: {str(e)}")
        return False, 0.0, f"Error: {str(e)}"


def preprocess_image(
    image_path,
    target_size=(128, 128),
    apply_threshold=False,
    thresh_method='otsu',
    thresh_val=127,
    morph_ops=None,
    kernel_size=3,
    morph_iterations=1
):
    """
    Preprocess image and optionally produce a binary mask using thresholding
    and morphological operations.

    Returns: (img_normalized_rgb, img_resized_bgr, mask)
      - img_normalized_rgb: RGB image normalized to 0..1 (numpy float32)
      - img_resized_bgr: resized BGR image as read by OpenCV (uint8)
      - mask: binary mask (uint8 0/255) or None if not requested

    Parameters:
      - apply_threshold: if True, compute a binary mask from grayscale image
      - thresh_method: 'otsu' or 'binary'
      - thresh_val: threshold value used when thresh_method=='binary'
      - morph_ops: list of operations to apply in order. Supported: 'erode',
        'dilate', 'opening', 'closing'
      - kernel_size: size of structuring element (odd integer)
      - morph_iterations: number of iterations for erosion/dilation
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if OpenCV fails
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Resize image (uses width, height tuple)
        img_resized = cv2.resize(img, target_size)

        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_normalized = img_rgb / 255.0

        mask = None
        if apply_threshold or (morph_ops is not None and len(morph_ops) > 0):
            # Convert to grayscale for thresholding/morphology
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Optionally apply threshold
            if apply_threshold:
                if thresh_method == 'otsu':
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, mask = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
            else:
                # start from simple binary of gray (non-thresholded) if morph ops requested
                _, mask = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)

            # Apply morphological operations if requested
            if morph_ops:
                # Ensure kernel size is odd and >=1
                ks = int(kernel_size)
                if ks <= 0:
                    ks = 1
                if ks % 2 == 0:
                    ks += 1

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

                for op in morph_ops:
                    op_lower = op.lower()
                    if op_lower == 'erode':
                        mask = cv2.erode(mask, kernel, iterations=morph_iterations)
                    elif op_lower == 'dilate':
                        mask = cv2.dilate(mask, kernel, iterations=morph_iterations)
                    elif op_lower == 'opening':
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                    elif op_lower == 'closing':
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
                    else:
                        # ignore unknown ops but continue
                        continue

        return img_normalized, img_resized, mask

    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {str(e)}")


def preprocess_pipeline(
        image_path,
    target_size=(128, 128),
        apply_threshold=True,
        thresh_method='otsu',
        thresh_val=127,
):
    """
    Simple preprocessing pipeline:
        1. Resize gambar
        2. Threshold / binarisasi
        3. Ekstraksi fitur RGB dan GLCM
    
        Returns: (img_resized_rgb, gray_processed)
            - img_resized_rgb: RGB image untuk fitur RGB (uint8)
            - gray_processed: grayscale hasil threshold untuk fitur GLCM (uint8)
    """
    try:
        # Step 1: Baca image
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if OpenCV fails
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise ValueError(f"Gagal membaca image dari {image_path}")
        
        # Step 2: Resize image ke target size (default 128×128)
        img_resized = cv2.resize(img, target_size)
        
        # Step 3: Threshold pada grayscale untuk memisahkan objek dari background
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        if apply_threshold:
            if thresh_method == 'otsu':
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
        else:
            mask = np.full(gray.shape, 255, dtype=np.uint8)

        # Terapkan mask ke RGB dan grayscale supaya fitur hanya membaca area relevan
        masked_bgr = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        img_resized_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
        gray_processed = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Validate outputs
        if img_resized_rgb is None:
            raise ValueError("img_resized_rgb is None after conversion")
        if gray_processed is None:
            raise ValueError("gray_processed is None after thresholding")
        
        return img_resized_rgb, gray_processed
        
    except Exception as e:
        print(f"[ERROR] preprocess_pipeline failed for {image_path}: {str(e)}")
        raise ValueError(f"Error in preprocessing pipeline {image_path}: {str(e)}")