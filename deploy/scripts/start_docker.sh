#start_docker.sh
#!/bin/bash
set -e
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to correct ECR..."
aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 546327345928.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling latest image..."
sudo docker pull 546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

echo "Stopping old container if exists..."
sudo docker stop home_price_api || true
sudo docker rm home_price_api || true

echo "Starting new container..."
sudo docker run -d -p 8000:8000 --name home_price_api \
-e DAGSHUB_USER_TOKEN=6e6324995eb1210c132b0a2cc13090484432f8e1 \
546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

echo "Container started successfully."
