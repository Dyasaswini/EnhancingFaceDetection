import cv2

# Load the image
image_path = "image.jpeg"
image = cv2.imread(image_path)

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Load the accessories (sunglasses and hat) images with transparent background
sunglasses_path = "glasses1.png"
#hat_path = "hat.png"
sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
#hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)

# Loop over the detected faces and add accessories
for (x, y, w, h) in faces:
    # Resize the accessories to fit the face
    sunglasses_resized = cv2.resize(sunglasses, (w, int(0.5 * h)))
    #hat_resized = cv2.resize(hat, (w, int(0.5 * h)))

    # Calculate the position to overlay the accessories
    sunglasses_y = y + int(0.35 * h)
    #hat_y = y - int(0.2 * h)

    # Overlay sunglasses
    for i in range(sunglasses_resized.shape[0]):
        for j in range(sunglasses_resized.shape[1]):
            if sunglasses_resized[i, j, 3] != 0:  # Check alpha channel for transparency
                image[sunglasses_y + i, x + j, :] = sunglasses_resized[i, j, :3]

    # Overlay hat
    #for j in range(hat_resized.shape[1]):
            #if hat_resized[i, j, 3] != 0:  # Check alpha channel for transparency
                #image[hat_y + i, x + j, :] = hat_resized[i, j, :3]

# Show the final image with accessories
cv2.imshow("Cool Accessories", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
