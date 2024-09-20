!sudo apt install git build-essential
!git clone https://github.com/Ronan-H/md5.git
%cd /content/md5
!ls
!gcc md5.c main.c -o md5 -lm
!./md5
