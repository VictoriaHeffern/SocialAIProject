import cv2
import numpy as np
import os

# Models for Facial Detection
# NOTE: All pre-trained models are Open source for academic purposes
# TODO: Should there be any future lisensing complicsations, lisence/train substitutes
detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Model for Facial Embedding (Face <-> Names)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Helper function to retreive an embedding from an image
def get_embedding(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (96, 96),
                                 (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob)
    return embedder.forward()

# init Known faces
known_embeddings = []
known_names = []

# Loop through known faces in face directory (Now just one img for Person)
# TODO: Implement functionality for multiple IMGs per individual
for file in os.listdir("faces"):
    path = os.path.join("faces", file)
    name = os.path.splitext(file)[0]
    img = cv2.imread(path)

    # detect face
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300))
    detector.setInput(blob)
    detections = detector.forward()

    if detections[0, 0, 0, 2] < 0.5:
        continue

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (x1, y1, x2, y2) = box.astype("int")
    face = img[y1:y2, x1:x2]

    embedding = get_embedding(face)

    known_embeddings.append(embedding.flatten())
    known_names.append(name)

print("Loaded all face embeddings.")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cap = cv2.VideoCapture(0)

# Loop for facial detection linked to computer Camera
while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
    detector.setInput(blob)
    detections = detector.forward()

    # Loop through each face detected on screen
    for i in range(0, detections.shape[2]):
        # confidence scoreing
        conf = detections[0, 0, i, 2]
        if conf < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        face = frame[y1:y2, x1:x2]
        embedding = get_embedding(face)
        embedding = embedding.flatten()

        # Compare to known embeddings
        scores = [cosine(embedding, kb) for kb in known_embeddings]
        best = np.argmax(scores)
        name = known_names[best]
        score = scores[best]

        cv2.putText(frame, f"{name} ({score:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
