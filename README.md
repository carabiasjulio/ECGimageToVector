# python-opencv-docker
Docker Image for Python + OpenCV 

# How to Run
docker build -t ecg_python_container .
docker run -it -v <ECG_IMAGES_HOST_PATH>:/home/ecgtovector/input ecg_python_container

# Example:
docker run -it -v /Users/carabias/PycharmProjects/ECG/ecgFolder:/home/ecgtovector/input ecg_python_container