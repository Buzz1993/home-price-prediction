#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 891377050051.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=home_price_api)" ]; then
    echo "Stopping existing container..."
    docker stop home_price_api
fi

if [ "$(docker ps -aq -f name=home_price_api)" ]; then
    echo "Removing existing container..."
    docker rm home_price_api
fi

echo "Starting new container..."
docker run -d -p 8000:8000 --name home_price_test -e DAGSHUB_USER_TOKEN=6e6324995eb1210c132b0a2cc13090484432f8e1 546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

echo "Container started successfully."
