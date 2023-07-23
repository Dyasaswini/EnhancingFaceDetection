
import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if cap.isOpened():
    # Loop over frames from the camera
    while True:
        # Capture a frame from the camera
        ret, image = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            face_roi = image[y:y+h, x:x+w]

            # Apply Gaussian blur to the face ROI
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)

            # Replace the original face ROI with the blurred face
            image[y:y+h, x:x+w] = blurred_face


        
        # Display the frame with the detected faces
        cv2.imshow('frame', image)

        # Check if the user has pressed the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Error: Failed to open camera.")
