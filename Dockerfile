
# pull base image
FROM patavee/scipy-matplotlib-py3
MAINTAINER Patavee Charnvivit <patavee@gmail.com>

# install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    cmake \
    pkg-config \
    clang \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libjasper-dev \
    libopenexr-dev \
    libgdal-dev \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libunicap2-dev \
    libv4l-0 \
    libv4l-dev \
    libxine2-dev \
    v4l-utils \
    libeigen3-dev \
    libtbb-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# define compilers
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

# compile
RUN cd /tmp && \
    wget -q -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip && \
    unzip -q opencv.zip && \
    wget -q -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip && \
    unzip -q opencv_contrib.zip && \
    mkdir opencv-3.2.0/build && \
    cd opencv-3.2.0/build && \
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_opencv_python2=NO \
    -D BUILD_JPEG=YES \
    -D WITH_WEBP=YES \
    -D WITH_OPENEXR=YES \
    -D BUILD_TESTS=NO \
    -D BUILD_PERF_TESTS=NO \
    .. && \
    make -j 4 VERBOSE=1 && \
    make && \
    make install