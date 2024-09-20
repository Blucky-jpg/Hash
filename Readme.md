1. Install prerequisites (git and gcc)  
   ```bash
   !sudo apt install git build-essential
2. Clone this repo 
   ```bash
   !git clone https://github.com/Blucky-jpg/AVHash.git
   %cd /content/AVHash
3. Compile  
   ```bash
   !gcc md5.c main.c -o md5 -lm
4. Run 
   ```bash
   !./md5
