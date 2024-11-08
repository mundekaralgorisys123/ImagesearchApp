#=============================================================================================================
                                        # ALL IMPORT HERE
#=============================================================================================================
import random
import numpy as np
import os
import openai
from flask import request, Flask, render_template,abort,flash,redirect,jsonify
from PIL import Image
from datetime import datetime 
from dotenv import load_dotenv
from io import BytesIO
import base64
import json
from prodia_client import Prodia, image_to_base64, validate_params 
from pathlib import Path
from werkzeug.utils import secure_filename
from pypdf import PdfReader 
from resumeparser import ats_extractor
from rsu import extract_text_from_pdf,get_openai_answer
import os
import time
from pathlib import Path
from threading import Thread
from sklearn.metrics.pairwise import cosine_similarity
from model.EfficientNetB0 import FeatureExtractor_EfficientNetB0
from model.ResNet50 import FeatureExtractor_ResNet50
from model.VGG19 import FeatureExtractor_VGG19
import cv2
from PIL import Image
from extractorcard import extract_data_from_image

#=============================================================================================================
load_dotenv()
#=============================================================================================================
                                        # CREATE APP CONSTRUCTOR HERE
#=============================================================================================================
# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
app.secret_key = 'your_secret_key'
#=============================================================================================================
                                        # Define upload and feature folders
#=============================================================================================================
UPLOAD_FOLDER = './static/uploads/' 
UPLOAD_FOLDER_BASE = './static/UploadImages/'
FEATURE_FOLDER_BASE = "./features"
training_status = {"status": "idle", "progress": 0}
features, img_paths = [], []  # Track extracted features and their image paths globally


FEATURE_FOLDER_BASE = ''
def feature_path_initialiser(feature_folder):
    global FEATURE_FOLDER_BASE
    FEATURE_FOLDER_BASE = './static/feature/'+feature_folder.lower()
    if not os.path.exists(FEATURE_FOLDER_BASE):
        os.makedirs(FEATURE_FOLDER_BASE)

if not os.path.exists(UPLOAD_FOLDER_BASE):
    os.makedirs(UPLOAD_FOLDER_BASE)
#=============================================================================================================

# Initialize Prodia API client with the API key
prodia_client = Prodia(api_key=os.getenv('PRODIA_API_KEY'))
model_names = prodia_client.list_models()

# Function to read the selected model from model.txt and set 'fe' globally
def get_selected_model():
    if not os.path.exists('model.txt'):
        return 'EfficientNetB0'  # Default model
    with open('model.txt', 'r') as file:
        return file.read().strip()
    
#=============================================================================================================
                                        # Initialize the Feature Extractor based on model selection
#=============================================================================================================   
fe = None
def initialize_extractor(selected_model):
    global fe
    if selected_model == 'EfficientNetB0':
        fe = FeatureExtractor_EfficientNetB0()
    elif selected_model == 'ResNet50':
        fe = FeatureExtractor_ResNet50()
    elif selected_model == 'VGG19':
        fe = FeatureExtractor_VGG19()

def get_folder_names(path):
    folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folder_names

def feature_initializer(FEATURE_FOLDER_BASE):
    global features
    global img_paths
    img_feature = []
    for feature_path in Path(FEATURE_FOLDER_BASE).glob("*.npy"):  # Loop through all saved features
        img_feature.append(np.load(feature_path))  # Load feature vector from .npy file
        img_paths.append(Path("./static/featureimg") / (feature_path.stem + ".jpg"))  # Create path for the corresponding image
    features = np.array(img_feature)  # Convert feature list to a NumPy array for vectorized operations
feature_initializer(FEATURE_FOLDER_BASE)


# Convert image to base64 string
def image_to_base64_str(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
#=============================================================================================================
                                    # Generate image from text prompt using Prodia API
#============================================================================================================= 

def txt2img(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed
    }
    
    validate_params(params)
    
    print("Sending request with parameters:", json.dumps(params, indent=4))
    result = prodia_client.generate(params)
    job = prodia_client.wait(result)
    
    return job["imageUrl"]


def img2img(input_image, denoising, prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
    params = {
        "imageData": image_to_base64(input_image),
        "denoising_strength": denoising,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed
    }
    
    validate_params(params)
    
    print("Sending request with parameters:", json.dumps(params, indent=4))
    result = prodia_client.transform(params)
    job = prodia_client.wait(result)
    
    return job["imageUrl"]


def process_image(input_image):
    img = Image.open(input_image)
    img = img.convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG', optimize=True, quality=85)
    if output_io.tell() > 4 * 1024 * 1024:
        raise ValueError("Compressed image is still larger than 4 MB")
    output_io.seek(0)
    return output_io


#=============================================================================================================
                                    # If feature.txt doesn't exist, return an empty list
#============================================================================================================= 

def get_features():
    if not os.path.exists('feature.txt'):
        return []
    with open('feature.txt', 'r') as file:
        features = [line.strip() for line in file.readlines() if line.strip()]
    if not features:
        return []
    return features


def get_models():
    if not os.path.exists('model.txt'):
        return []
    with open('model.txt','r') as file:
        models = [line.strip() for line in file.readlines() if line.strip()]
    if not models:
        return []
    return models

#=============================================================================================================
                                    # ROUTER START HERE MAIN PAGE
#============================================================================================================= 

@app.route('/')
def index():
    features = get_features()
    if not features:
        message = "No features are available to be used"
    else:
        message = None
    return render_template('index.html', features=features, message=message)

#=============================================================================================================
                                    # IMAGE GENERATE FEATURE
#============================================================================================================= 

@app.route('/generate', methods=['POST','GET'])
def generate_images():
    features = get_features()
    if not features:
        message = "No features are available to be used"
    else:
        message = None
    if request.method == 'POST':
        try:
            model = request.form['model']
            prompt = request.form['prompt']
            negative_prompt = request.form.get('negative-prompt', '')
            steps = request.form.get('steps', 30)
            sampling_method = request.form.get('sampling-method', 'default_sampler')
            cfg_scale = request.form.get('cfg-scale', 7.0)
            width = request.form.get('width', 512)
            height = request.form.get('height', 512)
            seed = request.form.get('seed', None)
            number_of_images = request.form['number-of-images']
            input_image = request.files.get('input-image', None)
            generated_images = []
            image_urls = []
            if input_image and input_image.filename != '':
                input_image = Image.open(input_image.stream).convert("RGBA")
                for i in range(int(number_of_images)):
                    current_seed = int(seed) + i * 5 if seed else random.randint(0, 10000)
                    result = img2img(
                        input_image=input_image,
                        denoising=float(request.form.get('denoising-strength', 0.5)),
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=model,
                        steps=int(steps) + i * 2,
                        sampler=sampling_method,
                        cfg_scale=float(cfg_scale),
                        height=int(height),
                        width=int(width),
                        seed=current_seed
                    )
                    generated_images.append(result)
                input_image = request.files.get('input-image', None)
                number_of_images = int(number_of_images)
                if number_of_images > 5:
                    number_of_images = 5
                response = openai.Image.create_variation(
                    model="dall-e-2",
                    image=process_image(input_image.stream),
                    n=int(number_of_images),
                    size="1024x1024"
                )
                image_urls = [image['url'] for image in response['data']]
            else:
                for i in range(int(number_of_images)):
                    current_seed = int(seed) + i * 5 if seed else random.randint(0, 10000)

                    result = txt2img(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model=model,
                        steps=int(steps) + i * 2,
                        sampler=sampling_method,
                        cfg_scale=float(cfg_scale),
                        height=int(height),
                        width=int(width),
                        seed=current_seed
                    )
                    generated_images.append(result)

                number_of_images = int(number_of_images)
                if number_of_images > 5:
                    number_of_images = 5
                response = openai.Image.create(
                    model="dall-e-2",
                    prompt=prompt,
                    size="1024x1024",
                    n=int(number_of_images)
                )
                image_urls = [image['url'] for image in response['data']]
            return render_template('result.html', generated_images=generated_images, image_urls=image_urls,features=features,message=message)
        except ValueError as ve:
            return f"Invalid input: {str(ve)}", 400
        except Exception as e:
            return f"An error occurred: {str(e)}", 500
    else:
        return render_template('generate.html',sampling_method=prodia_client.list_samplers(), model_names=model_names,features=features,message=message)

#=============================================================================================================
                                    # IMAGE SEARCH FEATURE
#============================================================================================================= 

@app.route('/search', methods=['GET', 'POST'])
def search_image():
    features2 = get_features()
    models = get_models()
    if not features2:
        message = "No features are available to be used"
    else:
        message = None
    if request.method == 'POST': 
        file = request.files['query_img']  
        img = Image.open(file.stream) 
        num_of_images = request.form.get('slider')
        num_of_images = int(num_of_images)
        model_name = request.form.get('modelName')
        feature_path_initialiser(model_name)
        FEATURE_FOLDER_BASE = './static/feature/'+model_name.lower()
        feature_initializer(FEATURE_FOLDER_BASE=FEATURE_FOLDER_BASE)
        initialize_extractor(model_name)
        global fe
        query = fe.extract(img)
        cos_sim = cosine_similarity([query], features)[0]
        ids = np.argsort(-cos_sim)[1:num_of_images+1] 
        scores = [(cos_sim[id], img_paths[id]) for id in ids]
        return render_template('search.html', query_path=image_to_base64_str(img), scores=scores,features=features2,models=models,model_name=model_name)  # Pass base64 image string
    else:
        return render_template('search.html', query_path=None, scores=None,features=features2,message=message,models=models,model_name = None)  # Render the default homepage on GET request

#=============================================================================================================
                                    # RESUME EXTRACTION
#============================================================================================================= 

def _read_file_from_path(path):
    reader = PdfReader(path) 
    data = ""
    for page_no in range(len(reader.pages)):
        page = reader.pages[page_no] 
        data += page.extract_text()
    return data 


@app.route('/resume', methods=['GET', 'POST'])
def resume():
    features = get_features()
    if not features:
        message2 = "No features are available to be used"
    else:
        message2 = None
    if request.method == 'POST':
        if 'query_pdf' not in request.files:
            return render_template('Resume.html', error="No file part")
        new_file = request.files['query_pdf'] 
        print(new_file)
        if new_file.filename == '':
            return render_template('Resume.html', error="No selected file")
        if new_file and new_file.filename.endswith('.pdf'):
            pdf_path = os.path.join(UPLOAD_FOLDER, new_file.filename)
            new_file.save(pdf_path) 
            print(pdf_path)
            data = _read_file_from_path(pdf_path)
            extracted_data = ats_extractor(data)
            data=json.loads(extracted_data)
            print(data)
            return render_template('Resume.html', message="File uploaded successfully!", data=data,features=features)
        else:
            return render_template('Resume.html', error="Please upload a valid PDF file",features=features)
    return render_template('Resume.html',features=features,message2=message2) 


#=============================================================================================================
                                    # RESUME EXTRACTION AND ASK QUESTIONS
#============================================================================================================= 


@app.route('/Research')
def Research():
    features = get_features()
    if not features:
        message = "No features are available to be used"
    else:
        message = None
    return render_template('Research.html',features=features,message=message)


@app.route('/askbot', methods=['POST'])
def askbot():
    features = get_features()
    if request.method == 'POST':
        uploaded_file = request.files.get('resume')
        search_phrase = request.form.get('question')
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)
            extracted_text = extract_text_from_pdf(file_path)
            if extracted_text:
                answer = get_openai_answer(extracted_text, search_phrase)
                print(search_phrase)
                print(answer)
                return render_template('Research.html', question=search_phrase, answer=answer,features=features)
            else:
                return "Could not extract text from the PDF.", 400
        else:
            return "Invalid file type. Please upload a PDF.", 400
    return render_template('Research.html',features=features)

#=============================================================================================================
                                # UPLOAD IMAGE IN FOLDER FOR IMAGE SEARCH APP
#============================================================================================================= 

@app.route('/upload_images', methods=['GET', 'POST'])
def upload_images():
    features = get_features()
    models = get_models()
    message1 = None if features else "No features are available to be used"
    return render_template('upload.html', features=features, message1=message1,models=models)

@app.route('/upload', methods=['POST'])
def upload_im():
    try:
        folder_name = request.form.get('folder_name')
        files = request.files.getlist('images[]')
        
        global fe
        model_name = request.form.get('modelName')  
        feature_path_initialiser(model_name)
        initialize_extractor(model_name)
        feature_path_initialiser(model_name)
        
        if not folder_name:
            return jsonify({'error': 'Folder name is required!'}), 400

        folder_path = os.path.join(UPLOAD_FOLDER_BASE, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for file in files:
            if file and file.filename:
                file_path = os.path.join(folder_path, file.filename)
                file.save(file_path)
                file.seek(0) 
                file_content = BytesIO(file.read()) 
                file_path1 = os.path.join("./static/featureimg/", file.filename) 
                with open(file_path1, 'wb') as f:
                    f.write(file_content.getvalue())
    
        return jsonify({'message': 'Images uploaded successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def model_training_task(folder_name):
    global training_status
    training_status['status'] = 'in-progress'
    training_status['progress'] = 0

    folder_path = os.path.join(UPLOAD_FOLDER_BASE, folder_name)
    image_paths = sorted(Path(folder_path).glob("*.jpg"))
    total_images = len(image_paths)
    
    if total_images == 0:
        training_status['status'] = 'completed'
        training_status['progress'] = 100
        return

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"Processing image: {img_path}")
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path(FEATURE_FOLDER_BASE) / (img_path.stem + ".npy")
        np.save(feature_path, feature)
        
        # Calculate progress
        progress_percentage = (idx / total_images) * 100
        training_status['progress'] = progress_percentage
        print(f"Progress: {progress_percentage:.2f}%")
    
    training_status['status'] = 'completed'
    training_status['progress'] = 100
    print(f"Training completed on folder: {folder_name}")

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    folder_name = data.get('folder_name')
    if not folder_name:
        return jsonify({'error': 'Folder name is required!'}), 400
    folder_path = os.path.join(UPLOAD_FOLDER_BASE, folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder does not exist!'}), 400
    thread = Thread(target=model_training_task, args=(folder_name,))
    thread.start()
    return jsonify({'message': 'Model training started.'}), 200

@app.route('/training_status', methods=['GET'])
def training_status_endpoint():
    global training_status
    return jsonify(training_status)
#=============================================================================================================
                                # TRAIN SIMILAR IMAGE FOLDER SEARCH APP
#============================================================================================================= 
training_status1 = {'status': 'in-progress', 'progress': 0}
features = []
img_paths = []
@app.route('/train_similar', methods=['GET', 'POST'])
def train_similar():
    features = get_features()
    models = get_models()
    if not features:
        message1 = "No features are available to be used"
    else:
        message1 = None
    folder_names = get_folder_names('./static/UploadImages')
    return render_template('train_similar.html',features=features,message1=message1,folder_names=folder_names,models=models)

def model_training_task1(folder_name, force_retraining):
    global training_status1, features, img_paths
    training_status1['status'] = 'in-progress'
    training_status1['progress'] = 0
    
    folder_path = os.path.join(UPLOAD_FOLDER_BASE, folder_name)
    img_paths_in_folder = list(Path(folder_path).glob("*.jpg"))
    total_images = len(img_paths_in_folder)
    trained_images_count = 0
    
    for idx, img_path in enumerate(sorted(img_paths_in_folder), start=1):
        feature_path = Path(FEATURE_FOLDER_BASE) / (img_path.stem + ".npy")
        
        # Check if we should skip or retrain
        if feature_path.exists() and not force_retraining:
            print(f"Skipping {img_path.name} (feature already exists)")
            continue  # Skip if force_retraining is False
        
        # Retrain or train new files as required
        print(f"Training {img_path.name}")
        feature = fe.extract(Image.open(img_path))
        np.save(feature_path, feature)
        print(f"Feature saved at: {feature_path}")
        
        # Increment trained images count for accurate progress calculation
        trained_images_count += 1
        
        # Update progress based on total images needing training (retrained or new)
        progress_percentage = (trained_images_count / total_images) * 100
        training_status1['progress'] = progress_percentage
        print(f"Progress: {progress_percentage:.2f}%")

    # Finalize training status
    training_status1['status'] = 'completed'
    training_status1['progress'] = 100
    print(f"Training completed on folder: {folder_name}")
    
    # Reload features and paths
    global features
    features.clear()
    global img_paths
    img_paths.clear()
    for feature_path in Path(FEATURE_FOLDER_BASE).glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/featureimg") / (feature_path.stem + ".jpg"))
    features = np.array(features)

@app.route('/train-model', methods=['POST'])
def train():
    folder_name = request.form.get('folderName')
    model_name = request.form.get('modelName')
    force_retraining = request.form.get('force_retraining') == 'true'
    print("Force retraining:", force_retraining)
    print("Folder:", folder_name, "Model:", model_name)
    
    initialize_extractor(model_name)
    feature_path_initialiser(model_name)
    
    # Start model training in a separate thread
    thread = Thread(target=model_training_task1, args=(folder_name, force_retraining))
    thread.start()
    return jsonify({'message': 'Model training started.'}), 200

@app.route('/training-status', methods=['GET'])
def training_status_endpoint1():
    return jsonify(training_status1)

#=============================================================================================================
                                #Visiting card extraction
#============================================================================================================= 
@app.route('/card', methods=['GET', 'POST'])
def card():
    features = get_features()
    
    if not features:
        message1 = "No features are available to be used"
    else:
        message1 = None

    return render_template('card.html',features=features,message1=message1)


@app.route('/upload_card', methods=['POST'])
def upload_card():
    features = get_features()
    if request.method == 'POST':
        if 'query_image' not in request.files:
            return redirect(request.url)
        
        file = request.files['query_image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Open and process image
            input_image = Image.open(file)
            
            # Call the extractor function to get the data
            extracted_data = extract_data_from_image(input_image)
            
            # Pass results to the result page
            return render_template(
                'card.html',
                features=features,
                image=file.filename,
                website=extracted_data['website'],
                email=extracted_data['email'],
                pincode=extracted_data['pincode'],
                phone_numbers=extracted_data['phone_numbers'],
                address=extracted_data['address'],
                card_holder=extracted_data['card_holder'],
                company_name=extracted_data['company_name'],
                other_details=extracted_data['other_details']
            )
            

#=============================================================================================================
                                # Image to Sketch
#============================================================================================================= 

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def sharpen_edges(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(laplacian)
    return cv2.addWeighted(image, 1.5, sharpened, -0.5, 0)

def convert_to_high_quality_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast_enhanced = enhance_contrast(gray_image)
    sharpened_image = sharpen_edges(contrast_enhanced)
    inverted_image = cv2.bitwise_not(sharpened_image)
    blurred = cv2.GaussianBlur(inverted_image, (15, 15), sigmaX=10, sigmaY=10)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch_image = cv2.divide(sharpened_image, inverted_blurred, scale=256.0)
    return sketch_image

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route('/sketch', methods=['GET', 'POST'])
def sketch():
    features = get_features()
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            # Convert the uploaded file to a numpy array
            original_image = Image.open(uploaded_file)
            image_array = np.array(original_image)

            # Convert to a high-quality sketch
            sketch_image = convert_to_high_quality_sketch(image_array)
            sketch_pil_image = Image.fromarray(sketch_image)

            # Convert both images to Base64 strings
            original_image_base64 = image_to_base64(original_image)
            sketch_image_base64 = image_to_base64(sketch_pil_image)

            return render_template(
                'sketch.html',
                original_image_data=original_image_base64,
                sketch_image_data=sketch_image_base64,
                features=features
            )
    return render_template('sketch.html',features=features)


#=============================================================================================================
                                # COMING SOON PROJECTS FEATURES
#============================================================================================================= 

if __name__ == '__main__':
    app.run(debug=True)
    # app.run("0.0.0.0")
