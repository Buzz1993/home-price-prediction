#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "Updating packages..."
sudo apt-get update -y

echo "Installing Docker..."
sudo apt-get install -y docker.io

echo "Starting Docker..."
sudo systemctl start docker
sudo systemctl enable docker

echo "Installing utilities..."
sudo apt-get install -y unzip curl

echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install

echo "Done."
