#!/bin/bash
set -e

echo "Stopping existing container if running..."

sudo docker stop home_price_api || true
sudo docker rm home_price_api || true

echo "Old container stopped."
