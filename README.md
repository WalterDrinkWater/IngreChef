# IngreChef
TARUMT FYP

# setup
pip install -r requirement.txt
pip install -r yolo/requirements.txt

# comment the following lines to avoid object detection model to load
def load_det_model():
    return torch.hub.load('./yolo', 'custom','./yolo/best.pt', source='local',trust_repo=True)

det_model = load_det_model()