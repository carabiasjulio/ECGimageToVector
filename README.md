# python-opencv ECG to Signal Vector 
Docker Image for ECG to Signal Vector 


# How to build image and run (In case edit code ....)
docker build -t ecg_python_image .

docker run -it -v <ECG_IMAGES_HOST_PATH>:/home/ecgtovector/input ecg_python_image


# How to download pre-build image and run (just run out the box)
docker pull carabiasjulio/ecg2vector:latest

docker run -it -v <ECG_IMAGES_HOST_PATH>:/home/ecgtovector/input carabiasjulio/ecg2vector


# Run Example:
docker run -it -v /Users/carabias/PycharmProjects/ECG/ecgFolder:/home/ecgtovector/input ecg_python_image
