ARG CUDA_VER
ARG UBUNTU_VER

FROM nvidia/cuda:${CUDA_VER}-cudnn8-devel-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND="noninteractive"

# ssh, remote debug, python
RUN apt-get update -y \
 && apt-get install -y \
      ssh \
      libssl-dev \
      rsync \
      wget \
      git \
      python3 \
      python3-pip \
      sudo \
 && apt-get clean \
 && pip install --upgrade pip \
 && pip install wheel setuptools
 
# matplotlib dependencies   
RUN apt-get update -y \
 && apt-get install -y \
      libfontconfig1-dev \
      libagg-dev \
      libpng-dev \
      libqhull-dev \
      ffmpeg \
      libsm6 \
      libxext6 \
 && apt-get clean
 
# ssh-server
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'X11Forwarding yes'; \
    echo 'X11DisplayOffset 10'; \
    echo 'X11UseLocalhost no'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_ide \
  && mkdir /run/sshd

# project dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# user
RUN useradd -m -s /bin/bash yolo \
 && yes password | passwd yolo \
 && usermod -aG sudo yolo

WORKDIR /repo
RUN chown -R yolo /repo

# UID and GID of localhost are passed when running a container
ENV HOST_UID=""
ENV HOST_GID=""
COPY docker/set_yolo_uid_gid_and_run_ssh_server.sh /tmp

CMD ["/tmp/set_yolo_uid_gid_and_run_ssh_server.sh"]

