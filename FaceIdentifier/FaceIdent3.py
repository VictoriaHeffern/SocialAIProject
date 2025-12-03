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
    return embedder.forward().flatten()

# Helper function to detect single faces in an image
def detect_face(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300))
    detector.setInput(blob)
    detections = detector.forward()

    if detections.shape[2] == 0:
        return None

    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    if confidence < 0.75:
        return None
    
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)
    return image[y1:y2, x1:x2]

# init Known faces
def load_embedding(relative_face_path):
    known_embeddings = []
    known_names = []

    face_loc = relative_face_path

# Loop through and pull known faces from profiles
    for person in os.listdir(face_loc):
        person_path = os.path.join(face_loc, person)
        # If person does not exist
        if not os.path.isdir(person_path):
            continue

        print(f"Loading images from {person}'s profile")

        for img_name in os.listdir(person_path):
            image_path = os.path.join(person_path, img_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            face = detect_face(image)
            if face is None:
                continue

            embedding = get_embedding(face)

            known_embeddings.append(embedding)
            known_names.append(person)

        print("Loaded embeddings for all known profiles")
        print("Loaded:", known_names)
        print("Embeddings:", len(known_embeddings))
    
    return known_embeddings, known_names

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Loop for facial detection linked to computer Camera

def ident_in_frame(cap, embeddings, names):

    ret, frame = cap.read()
    if not ret: return None

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
    detector.setInput(blob)
    detections = detector.forward()

    # Loop through each face detected on screen, save largest bbox_area and face
    best_area = 0
    best_name = "NONE"
    best_name_conf = -1
    detection_conf = -1
    for i in range(0, detections.shape[2]):
        # confidence scoreing
        conf = detections[0, 0, i, 2]
        if conf < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        embedding = get_embedding(face)

        # Compare to known embeddings
        scores = [cosine(embedding, kb) for kb in embeddings]
        best = np.argmax(scores)
        name = names[best]
        score = scores[best]

        if score < 0.75:
            name = "Unknown"
        this_area = bbox_area(x1, x2, y1, y2)
        if this_area > best_area:
            best_area = this_area
            best_name = name
            best_name_conf = score
            detection_conf = conf
        cv2.putText(frame, f"{name} ({score:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == 27: return

    return best_name, best_name_conf, detection_conf

def bbox_area(x1, x2, y1, y2):
    width = np.abs(x1 - x2)
    height = np.abs(y1 - y2)

    return width * height



### EXAMPLE CODE FOR MAIN FILE

cap = cv2.VideoCapture(0)
face_dir = "faces"
embeddings, names = load_embedding(face_dir)

while True:
    # identity => name of the person in frame, unkown if not known, NONE if no face in frame
    # ident_conf => confidence that the face in frame is this person/unknown, -1 if no face in frame
    # face_conf => confidence that the thing in the bounding box is a face, -1 if no face (below detection threshold)
    identity, ident_conf, face_conf = ident_in_frame(cap, embeddings, names)
    if identity is not None:
        print(identity + f", {ident_conf}, {face_conf}")

# while True:
#     ret, frame = cap.read()
#     if not ret: break
# 
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
#     detector.setInput(blob)
#     detections = detector.forward()
# 
#     
# 
#     # Loop through each face detected on screen
#     for i in range(0, detections.shape[2]):
#         # confidence scoreing
#         conf = detections[0, 0, i, 2]
#         if conf < 0.6:
#             continue
# 
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (x1, y1, x2, y2) = box.astype("int")
# 
#         face = frame[y1:y2, x1:x2]
#         if face.size == 0:
#             continue
# 
#         embedding = get_embedding(face)
# 
#         # Compare to known embeddings
#         scores = [cosine(embedding, kb) for kb in known_embeddings]
#         best = np.argmax(scores)
#         name = known_names[best]
#         score = scores[best]
# 
#         if score < 0.75:
#             name = "Unknown"
# 
#         cv2.putText(frame, f"{name} ({score:.2f})",
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8, (0, 255, 0), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2),
#                       (0, 255, 0), 2)
# 
#     cv2.imshow("Recognition", frame)
#     if cv2.waitKey(1) == 27: break
# 
# cap.release()
# cv2.destroyAllWindows()
