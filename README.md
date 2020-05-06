# python-opencv ECG to Signal Vector 
Docker Image for ECG to Signal Vector 

# How to Run
docker build -t ecg_python_image .
docker run -it -v <ECG_IMAGES_HOST_PATH>:/home/ecgtovector/input ecg_python_image

# Example:
docker run -it -v /Users/carabias/PycharmProjects/ECG/ecgFolder:/home/ecgtovector/input ecg_python_image