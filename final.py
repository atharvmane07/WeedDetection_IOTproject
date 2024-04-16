import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('crop_weed_detection_model.h5')

def browse_video():
    global video_path
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video()

def process_video():
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize, normalize, etc.)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        # Make a prediction
        predictions = model.predict(np.expand_dims(frame, axis=0))
        weed_detected = predictions[0][0] >= 0.5  # You can adjust the threshold here

        # Display the result on the frame
        if weed_detected:
            result_text = "Weed Detected"
            frame = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            result_text = "No Weed Detected"
            frame = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            break
        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def browse_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        process_image()

def process_image():
    global image_label
    global result_label

    # Load and preprocess the selected image
    test_image = cv2.imread(image_path)
    if test_image is None:
        # Handle case where image loading fails
        print("Error loading image.")
        return

    # Check if the image is already in grayscale
    if len(test_image.shape) == 2:
        gray_image = test_image
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    original_image = cv2.resize(test_image, (224, 224))  # Resize the original image for display
    test_image = cv2.resize(test_image, (224, 224))
    test_image = test_image / 255.0

    # Make a prediction
    predictions = model.predict(np.expand_dims(test_image, axis=0))
    weed_detected = predictions[0][0] >= 0.5  # You can adjust the threshold here

    # Highlight the detected regions
    if weed_detected:
        # Threshold the image to create a binary mask
        _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw green square boxes around detected weeds
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Adjust coordinates to be within the original image dimensions
            x1, y1 = max(0, x - 10), max(0, y - 10)  # Top-left corner
            x2, y2 = min(original_image.shape[1], x + w + 10), min(original_image.shape[0], y + h + 10)  # Bottom-right corner
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert the processed image to a format compatible with Tkinter
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(processed_image)
        processed_image = ImageTk.PhotoImage(processed_image)

        # Display the processed image in the GUI
        image_label.config(image=processed_image)
        image_label.image = processed_image

        # Update the result label
        result_label.config(text="Weed Detected")
    else:
        # No weed detected, display the original image
        original_image = Image.fromarray(original_image)
        original_image = ImageTk.PhotoImage(original_image)

        image_label.config(image=original_image)
        image_label.image = original_image

        result_label.config(text="No Weed Detected")

# Create the main GUI window
root = tk.Tk()
root.title("Weed Detection from Video and Image")

# Create GUI elements
browse_video_button = tk.Button(root, text="Browse Video", command=browse_video)
browse_image_button = tk.Button(root, text="Browse Image", command=browse_image)
image_label = tk.Label(root)
result_label = tk.Label(root, font=("Helvetica", 16))

# Pack GUI elements
browse_video_button.pack()
browse_image_button.pack()
image_label.pack()
result_label.pack()

root.mainloop()






