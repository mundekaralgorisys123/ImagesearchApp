# Image Processing and Resume Management System

## Overview
This project implements the following functionalities:
1. **Image Search**: Upload images, extract features using pre-trained models, and store them for efficient image search.
2. **Image Generation**: Generate new images or variations of uploaded images using OpenAI's DALL-E API or similar.
3. **Resume Summary**: Extract and summarize key information from resumes such as work experience, skills, and education.
4. **Resume Search**: Search through uploaded resumes based on specific criteria like years of experience or skillsets.

## Table of Contents
- [Installation](#installation)
- [Image Search](#image-search)
- [Image Generation](#image-generation)
- [Resume Summary](#resume-summary)
- [Resume Search](#resume-search)
- [Technologies Used](#technologies-used)
- [Server Deployment](#server-deployment)
- [Nginx and Gunicorn Setup](#nginx-and-gunicorn-setup)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository.git
    cd your-repository
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the database (for image search and resume search) by running:
    ```bash
    python setup_database.py
    ```

## Image Search

### Process:
1. Upload images to the server.
2. Use the `offline.py` script to extract image features using a pre-trained model like ResNet or VGG.
3. Store the extracted features in a separate folder for future searches.

### Usage:
1. **Upload Images**: Upload images via the provided API or UI.
2. **Extract Features**: Run the feature extraction script:
    ```bash
    python offline.py --input_folder ./images --output_folder ./features
    ```
3. **Search**: Search for similar images based on the uploaded image using the feature database. The system compares embeddings to find similar images.

### Implementation:
- Uses pre-trained models (e.g., ResNet, VGG) for feature extraction.
- A feature database (FAISS or Elasticsearch) stores the embeddings for efficient search and retrieval.

## Image Generation

### Process:
1. Upload an image to generate variations.
2. The backend processes the image, passes it to OpenAI’s DALL-E API (or similar), and generates new images based on a specific prompt.
3. Return and display the generated images.

### Usage:
1. **Upload Image**: Upload an image via the UI.
2. **Generate Image**: Submit the prompt for image generation. The backend will call the DALL-E API and return a newly generated image.

### Implementation:
- Integrates DALL-E or similar models to create new images based on a prompt and the uploaded image.
- A UI allows users to upload images and enter prompts for generating similar or new designs.

## Resume Summary

### Process:
1. Upload a resume (PDF, DOCX, etc.).
2. Extract key information like work experience, skills, and education.
3. Generate a summarized version of the resume for easier review.

### Usage:
1. **Upload Resume**: Upload the resume via the UI or API.
2. **Extract and Summarize**: The system processes the resume, extracting important details and generating a summary.

### Implementation:
- Utilizes NLP models such as Spacy or Hugging Face for information extraction.
- Summarization algorithms are used to create short descriptions of experience and skills.

## Resume Search

### Process:
1. Upload resumes to a searchable database.
2. Search through resumes based on criteria like years of experience, specific skills, and education.

### Usage:
1. **Upload Resume**: Upload resumes to the database.
2. **Search Resumes**: Perform a search based on specific criteria using the search API or UI.

### Implementation:
- Resumes are stored in a structured format in a database (e.g., Elasticsearch, MongoDB).
- Search functionality allows filtering resumes based on relevant queries.

## Technologies Used
- **Backend**: Python, Flask/Django
- **Image Search**: OpenCV, Pre-trained Models (ResNet, VGG), FAISS/Elasticsearch
- **Image Generation**: OpenAI DALL-E API (or similar)
- **NLP for Resume Processing**: Spacy, Hugging Face Transformers
- **Database**: MongoDB, Elasticsearch
- **Frontend**: HTML/CSS/JavaScript (for UI)

## Server Deployment

### One-Time Setup for Nginx and Gunicorn

### Nginx Configuration

1. **Configure Nginx with the App (`ai_tools`)**:
   Create an Nginx configuration file at `/etc/nginx/sites-available/ai_tools` and add the following content:
   
   ```nginx
   server {
       listen 80;
       server_name your_domain_or_ip;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }


2. **Enable the site**:

   ```bash
   sudo ln -s /etc/nginx/sites-available/ai_tools /etc/nginx/sites-enabled
   ```

3. **Test Nginx Configuration**:

   ```bash
   sudo nginx -t
   ```

4. **Restart Nginx**:

   ```bash
   sudo systemctl restart nginx
   ```

5. **Allow Nginx through the firewall**:

   ```bash
   sudo ufw allow 'Nginx Full'
   ```

---

## Install and Configure Gunicorn

1. **Activate Your Virtual Environment**:

   ```bash
   cd /var/www/html
   source venv/bin/activate
   ```

2. **Install Gunicorn**:

   ```bash
   pip install gunicorn
   ```

3. **Run the Flask App Using Gunicorn**:

   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```

---

## Using `systemd` to Keep the Flask App Running

1. **Create a `systemd` Service File**:

   ```bash
   sudo nano /etc/systemd/system/flaskapp_image.service
   ```

2. **Add the Following Configuration**:

   ```ini
   [Unit]
   Description=Gunicorn instance to serve Flask app
   After=network.target

   [Service]
   User=www-data
   Group=www-data
   WorkingDirectory=/var/www/html
   Environment="PATH=/var/www/html/venv/bin"
   ExecStart=/var/www/html/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 app:app

   [Install]
   WantedBy=multi-user.target
   ```

3. **Reload `systemd` and Start the Flask App**:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start flaskapp_image
   sudo systemctl enable flaskapp_image
   ```

4. **Check the Status**:

   ```bash
   sudo systemctl status flaskapp_image
   ```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
