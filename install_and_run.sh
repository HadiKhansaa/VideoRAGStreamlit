#!/bin/bash

# Check if fusion_model.zip exists
if [ ! -f fusion_model.zip ]; then
    echo "Error: fusion_model.zip not found."
    echo "Please download the fusion_model.zip file from Colab first."
    exit 1
fi

# Create directory for application
echo "Creating application directory..."
mkdir -p video_rag_app
cd video_rag_app

# Extract fusion model
echo "Extracting fusion model..."
unzip -o ../fusion_model.zip

# Check if app.py exists
if [ ! -f app.py ]; then
    echo "Creating app.py..."
    # You should paste the Streamlit app code here
    echo "Error: Please create app.py manually or copy it from the instructions."
    echo "Place app.py in the video_rag_app directory."
    exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r fusion_model/requirements.txt

# Run the application
echo "Starting Streamlit application..."
streamlit run app.py