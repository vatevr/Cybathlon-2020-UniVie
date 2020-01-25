docker run -it -p 5901:5901 -p 6901:6901 --user 0:0 --privileged -v /dev/bus/usb:/dev/bus/usb neuroidss/openvibe-ubuntu-xfce-vnc:2.2.0-freeeeg32-alpha1.5 ./openvibe-2.2.0-src/dist/extras-Release/openvibe-acquisition-server.sh

connect via VNC viewer localhost:5901, default password: vncpassword connect via noVNC HTML5 full client: http://localhost:6901/vnc.html, default password: vncpassword connect via noVNC HTML5 lite client: http://localhost:6901/?password=vncpassword

Dockerfile sources: https://github.com/neuroidss/FreeEEG32-alpha1.5/blob/master/OpenVIBE/Dockerfile

Docker container autorun: openvibe-acquisition-server.sh, openvibe-designer.sh https://github.com/neuroidss/FreeEEG32-alpha1.5/blob/master/OpenVIBE/openvibe-ubuntu-xfce-vnc-2.2.0-freeeg32-alpha1.5.tar.xz
