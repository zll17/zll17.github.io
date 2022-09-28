# 1. on Ubuntu: create a new user account
#mkdir /home/zll
#useradd zll -d /home/zll -s /bin/bash
#chown -Rh zll /home/zll
#usermod -g root zll
#passwd zll
#echo "zll   ALL=(ALL)   ALL" >> /etc/sudoers
#ssh zll@localhost -y
# 2. on Ubuntu: ssh-keygen copy pub_id to my lap
ssh-keygen
cat ~/.ssh/id_rsa.pub 
# 3. modify apt source && apt-get update & upgrade
sudo sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

sudo apt-get update && sudo apt-get upgrade

# 4. install fundamental softwares
apt install curl -y

apt install vim wget git cron dnsutils unzip lrzsz fdisk gdisk exfat-fuse exfat-utils -y


# 4. install python3.6
sudo apt install software-properties-common

# 5. modify dash to bash

# 6. config vim
mkdir .vim
cd .vim
mkdir bundle
git clone https://github.com.cnpmjs.org/VundleVim/Vundle.vim.git
cd ~/
wget https://zll17.github.io/assets/zll.vimrc -O .vimrc

# 7. scp .bashrc to Ubuntu
wget https://zll17.github.io/assets/zll.bashrc -O .bashrc
wget https://zll17.github.io/assets/zll.bash_profile -O .bash_profile