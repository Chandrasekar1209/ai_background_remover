import cv2

def preprocess_image(image_path):
    # Read the image from the given file path
    img = cv2.imread(image_path)
    
    # Convert the image from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 512x512 (standard size for many models)
    img = cv2.resize(img, (512, 512))
    
    # Normalize the image by scaling pixel values to the range [0, 1]
    return img / 255.0
