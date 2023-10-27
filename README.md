# Multi-Person-Face-Recognition
The project is to create a trained ML model that can perform multi-person identification in the live video feed on NVIDIA A100 DGX Server. 


We have used popular DLib and face_recognition libraries as the basis of the project 
- https://github.com/ageitgey/face_recognition


## Creating Docker Image
- Clone the repository
- Change the directory to the cloned repository
- Run the following command to create the docker image
    1. On DGX Server
        - `docker build -t Multi-Person-Face-Recognition:latest .`
    2. On Local Machine
        - `docker build -t Multi-Person-Face-Recognition:latest -f Dockerfile.local .`

## Running the Docker Image

