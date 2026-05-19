# #install_dependencies.sh
# #!/bin/bash
# set -e
# export DEBIAN_FRONTEND=noninteractive

# echo "Updating packages..."
# sudo apt-get update -y

# echo "Installing Docker..."
# sudo apt-get install -y docker.io

# echo "Starting Docker..."
# sudo systemctl start docker
# sudo systemctl enable docker

# echo "Installing utilities..."
# sudo apt-get install -y unzip curl

# echo "Downloading AWS CLI..."
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
# unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/

# if ! command -v aws &> /dev/null
# then
#     echo "Installing AWS CLI..."
#     sudo /home/ubuntu/aws/install
# else
#     echo "AWS CLI already installed, skipping..."
# fi

# echo "Cleanup..."
# rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws

# echo "Dependencies installed successfully."


#-------------------------------------------------------------------------------------------------------------

#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# 🔁 Toggle (same idea as CI/CD)
ENABLE_AWS=false   # change to true when deploying to AWS

echo "Updating packages..."
sudo apt-get update -y

echo "Installing Docker..."
sudo apt-get install -y docker.io

echo "Starting Docker..."
sudo systemctl start docker
sudo systemctl enable docker

echo "Installing utilities..."
sudo apt-get install -y unzip curl

# 🔥 AWS CLI only if needed
if [ "$ENABLE_AWS" = true ]; then
    echo "Downloading AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"

    unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/

    if ! command -v aws &> /dev/null
    then
        echo "Installing AWS CLI..."
        sudo /home/ubuntu/aws/install
    else
        echo "AWS CLI already installed, skipping..."
    fi

    echo "Cleanup..."
    rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws
else
    echo "AWS setup skipped (ENABLE_AWS=false)"
fi

echo "Dependencies installed successfully."