
import torch
from commons import get_model,get_tensor


class_names=['BrownSpot', 
			 'Healthy', 
			 'Hispa', 
			 'LeafBlast']
model=get_model()
def get_flower_name(image_bytes):
    tensor=get_tensor(image_bytes)
    outputs=model(tensor)
    _,prediction=outputs.max(1)
    category=prediction.item()
    food_name=class_names[category]

    return food_name