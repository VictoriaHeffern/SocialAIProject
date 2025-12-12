# Instructions to run

## NAO_AnimatedServer
Due to the use of naoqi, this file must be run on a venv (virtual environment) running python 2.7

## Main
Main, along with its associated scripts/imports below, can be run on a 3.12 venv.

### FaceIdent3
Requires that you have a folder named "faces" in your working directory. This folder contains subfolders that are labeled with the names of users, where each named subfolder contains images of that user in jpeg format.  
Two additional files are provided in this folder: "deploy.prototxt" and "res10_300x300_ssd_iter_140000_fp16.caffemodel".  
Due to file sizes, we were unable to include the last model "openface_nn4.small2.v1.t7" as part of this repository, and it must be downloaded on your own and added to the working directory.

### AIConvoModel
Add a "MemoryDatabase" folder to your desktop. This folder should be populated with text files that are named with the names of users. The formatting of the text file should be:  
Name:  
Hobbies:  
Likes:  
Age:  
Occupation:  

### ClientServer
