# #start_docker.sh


# #!/bin/bash
# set -e
# exec > /home/ubuntu/start_docker.log 2>&1

# echo "Logging in to correct ECR..."
# aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 546327345928.dkr.ecr.ap-south-1.amazonaws.com

# echo "Pulling latest image..."
# sudo docker pull 546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

# echo "Stopping old container if exists..."
# sudo docker stop home_price_api || true
# sudo docker rm home_price_api || true

# echo "Starting new container..."
# sudo docker run -d -p 8000:8000 --name home_price_api \
# -e DAGSHUB_USER_TOKEN=875e26c303d4b004f67cb17f815ac0736597ca44 \
# 546327345928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest

# echo "Container started successfully."


#!/bin/bash

LOG_FILE="/home/ubuntu/start_docker.log"
exec > $LOG_FILE 2>&1

echo "==== DEPLOYMENT START ===="
date

echo "Checking Docker service..."
sudo systemctl start docker || true

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 54632745928.dkr.ecr.ap-south-1.amazonaws.com || {
  echo "ECR login failed"
}

echo "Pulling latest image..."
sudo docker pull 54632745928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest || {
  echo "Docker pull failed"
}

echo "Stopping old container if exists..."
sudo docker stop home_price_api || true
sudo docker rm home_price_api || true

echo "Starting new container..."
sudo docker run -d -p 8000:8000 --name home_price_api \
-e DASHUB_USER_TOKEN=875e26c303d4b004f67cb17f815ac0736597ca44 \
54632745928.dkr.ecr.ap-south-1.amazonaws.com/home_price_prediction:latest || {
  echo "Docker run failed"
}

echo "==== DEPLOYMENT END ===="
date

exit 0
