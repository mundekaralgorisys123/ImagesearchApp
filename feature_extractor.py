import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input  # Import VGG19 instead of VGG16
from tensorflow.keras.models import Model
import numpy as np

tensorflow.config.threading.set_inter_op_parallelism_threads(4)
tensorflow.config.threading.set_intra_op_parallelism_threads(4)

class FeatureExtractor:
    def __init__(self):
        # Load the pre-trained VGG19 model with weights trained on the ImageNet dataset.
        base_model = VGG19(weights="imagenet")  # Use VGG19 instead of VGG16
        
        # Extract features from the 'fc1' layer (the first fully connected layer).
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    def extract(self, img):
        img = img.resize((224, 224)).convert('RGB')  # Resize and convert image to RGB
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input
        x = preprocess_input(x)  # Preprocess input to match ImageNet expectations
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)  # Normalize the feature vector
