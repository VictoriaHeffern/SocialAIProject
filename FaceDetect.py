import cv2

# Load the Haar cascade face detection
#   - contains data about human features
#   - scans an image to determine where face is/if face is present
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# get one frame from webcam
ret, frame = cap.read()

# If frame read 
if ret:
    # Convert to greyscale for cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    #   detectMultiScale moves detection window across the image at 
    #           different scales to find faces of different sizes
    #   scaleFactor=1.1: The image is resized by 10% smaller each time 
    #           the detector scans for faces at a new scale (helps detect faces of various sizes).
    #   minNeighbors=5: Controls how many “neighboring rectangles” a detection 
    #           should have to be retained. Higher value = less false positives, might miss some faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(f"Detected {len(faces)} face(s)")

    cv2.imshow('Face Detection', frame)
    cv2.waitKey(0)  # Wait until you press a key
else:
    print("Failed to grab frame")

# Release the webcam and close
cap.release()
cv2.destroyAllWindows()
