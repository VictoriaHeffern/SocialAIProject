from FaceIdent3 import *
from AIConvoModel import *
import cv2
import numpy as np
from collections import Counter
import time
import os

cap = cv2.VideoCapture(0)
face_dir = "faces"
embeddings, names = load_embedding(face_dir)

# init variables
guess_list=[]
reading=[]
guess_names=[]
sum = 0

# init time stuff
elapsed_time = 0
start_time = time.perf_counter()


while elapsed_time < 10:
    # identity => name of the person in frame, unkown if not known, NONE if no face in frame
    # ident_conf => confidence that the face in frame is this person/unknown, -1 if no face in frame
    # face_conf => confidence that the thing in the bounding box is a face, -1 if no face (below detection threshold)
    identity, ident_conf, face_conf = ident_in_frame(cap, embeddings, names)
    if identity is not None:
        print(identity + f", {ident_conf}, {face_conf}")
        reading=[identity, ident_conf, face_conf]
        guess_list.append(reading)

    curr_time = time.perf_counter()
    elapsed_time = curr_time - start_time

cap.release()
cv2.destroyAllWindows()

# For every reading, remove those with small confidence values
for item in guess_list:
    if item[1]<0.8: # remove data with tiny num
        guess_list.remove(item)

# Create a list of the names
for item in guess_list:
    guess_names.append(item[0])
    
# Find the most common name
string_counts = Counter(guess_names) 
most_common_element_tuple= string_counts.most_common(1)
most_likley_user = most_common_element_tuple[0][0]
print("----------------------------------------")
print("The current face is most likley:", most_likley_user)

if most_likley_user != "Unknown":
    user_interaction_chat(True, most_likley_user)
else:
    user_interaction_chat(faceKnown=False)