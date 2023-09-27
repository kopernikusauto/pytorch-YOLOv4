#!/usr/bin/env bash

# set uid and gid to host ones 
# to get permissions to mounted volumes
usermod -u ${HOST_UID} yolo
groupmod -g ${HOST_GID} yolo

# run ssh server
/usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_ide

