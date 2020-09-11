import io
import json
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from skorch import NeuralNetClassifier


labels = json.load(open('deeplearning/dl_model/plant_classes.json'))

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

def load_model():
     """Load Pytorch model."""
     global model
     # Load and initialize the model
     model = NeuralNetClassifier(PretrainedModel, criterion=nn.CrossEntropyLoss, module__output_features=38)
     model.initialize()
     # load up the saved model weights
     model.load_params(f_params='deeplearning/dl_model/model_params.pt',
                       f_optimizer='deeplearning/dl_model/model_optimizer.pt',
                       f_history='deeplearning/dl_model/model_history.json')

# Load trained models
load_model()

def classify_image(image_bytes):
    '''
    Function to pre-process the input image to a format similar to training data and
    then make predictions on the image
    '''
    # define the various image transformations we have to make
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    # Open and identify the uploaded/given image file.
    image = Image.open(io.BytesIO(image_bytes))
    # Pre-process the uploaded image
    tensor = transform(image).unsqueeze(0)

    # Make predictions
    outputs = model.forward(tensor)
    _, indices = torch.sort(outputs, descending=True)
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Return the top 3 pedictions ('class label, probability')
    result = [ (labels[str(idx.item())][0], labels[str(idx.item())][1], confidence[idx].item()) for idx in indices[0][:5] ]

    return result
