import cv2
import numpy as np
from PIL import Image

def preprocess_image(
    image_path,
    target_size=(256, 256),
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