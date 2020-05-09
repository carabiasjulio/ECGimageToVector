FROM python:3.7
MAINTAINER Josip Janzic <josip@jjanzic.com>

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        poppler-utils \
        libcurl4-openssl-dev \
        libexpat1-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy matplotlib pdf2image wfdb

WORKDIR /
ENV OPENCV_VERSION="4.1.1"
RUN wget https://archive.physionet.org/physiotools/wfdb.tar.gz \
&& tar xfvz wfdb.tar.gz \
&& cd wfdb-10.6.2 \
&& ./configure \
&& make install \
&& make check

RUN wget -r -N -c -np https://physionet.org/files/ecgpuwave/1.3.4/ \
&& cd physionet.org/files/ecgpuwave/1.3.4/src/ecgpuwave/ \
&& make install \
&& make check

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.7) \
  -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}
RUN ln -s \
  /usr/local/python/cv2/python-3.7/cv2.cpython-37m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.7/site-packages/cv2.so

#copying helloworld.py from local directory to container's helloworld folder
COPY ecgImage.py /home/ecgProject/ecgImage.py
COPY ecgSignal.py /home/ecgProject/ecgSignal.py
COPY demo.py /home/ecgProject/demo.py
# COPY ecgc-set1 /home/ecgtovector/ecgc-set1

#running helloworld.py in container
CMD python /home/ecgProject/demo.py