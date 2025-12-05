from FaceIdent3 import load_embedding, ident_in_frame
from ClientServer import ConversationClient
import cv2
import time
from collections import Counter

server_host = '127.0.0.1'
server_port = 9000

def main():
    face_dir = "faces"
    embeddings, names = load_embedding(face_dir)

    cap = cv2.VideoCapture(0)

    guess_list = []
    start_time = time.perf_counter()
    elapsed_time = 0

    print("Detecting face for 10 seconds...")

    while elapsed_time < 10:
        identity, ident_conf, face_conf = ident_in_frame(cap, embeddings, names)
        if identity is not None and ident_conf >= 0.8:
            print(f"Detected: {identity}, Confidence: {ident_conf:.2f}")
            guess_list.append(identity)
        elapsed_time = time.perf_counter() - start_time

    cap.release()
    cv2.destroyAllWindows()

    if not guess_list:
        print("No confident face detected, proceeding as unknown user.")
        faceKnown = False
        person_name = None
    else:
        most_common_name = Counter(guess_list).most_common(1)[0][0]
        print(f"Most likely user: {most_common_name}")
        if most_common_name == "Unknown":
            faceKnown = False
            person_name = None
        else:
            faceKnown = True
            person_name = most_common_name

    client = ConversationClient(server_host, server_port, face_known=faceKnown, person_name=person_name)
    if not client.connect():
        print("Failed to connect to NAO server. Exiting.")
        return

    client.start_listening()

    while client.connected and client.listening:
        time.sleep(0.1)

    client.disconnect()

if __name__ == "__main__":
    main()

