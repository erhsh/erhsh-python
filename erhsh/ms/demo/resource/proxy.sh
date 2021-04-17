#!/bin/bash

read -r -p "Input proxy addr [proxy.huawei.com]: " addr
read -r -p "Input proxy port [8080]: " port
read -r -p "Input username: " username
read -r -s -p "Input password: " password

addr=${addr:-proxy.huawei.com}
port=${port:-8080}
username=$(echo "$username" | tr -d '\n' | xxd -plain | sed 's/\(..\)/%\1/g')
password=$(echo "$password" | tr -d '\n' | xxd -plain | sed 's/\(..\)/%\1/g')
proxy="http://$username:$password@$addr:$port"

echo -e "\n--------------"
echo -e "set success. proxy=" "$proxy"

export ftp_proxy=$proxy
export http_proxy=$proxy
export https_proxy=$proxy
export no_proxy=10.*,*.huawei.com,$no_proxy
