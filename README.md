# CV model to predict brand of beer 
- Model used: YOLO
- labelImg used to label images

- Rough outline:
1. label 50 images
2. test yolo
3. if works, train on all 500 images

- 80% training set, 20% validation
- labels: https://docs.google.com/document/d/1FpFuS3t5O0PdLV6cuKWcUKLSrLfABW0MR47Vb34X-bY/edit

### To download Yolo
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt  

### To download labelImg
without venv: (try this first)
- sudo pip3 install labelImg

with venv:

- creating virtual environment
    - python3 -m venv path/to/venv
    - source path/to/venv/bin/activate

### How to Label Images
1. in terminal, run `labelImg train1 label1/classes.txt`
2. format for selection should be 'YOLO'
3. 'w' to create RectBox, 'a' to go to previous image, 'd' for next image


### Running yolo

- To find image size: in main directory, run `python image_size.py`
    - the input for yolo model must be square, so if 
        Width: 600, Height: 800
      -- img 640

1. make sure dataset directory contains correct images and labels
- labels are .txt files created after labelling using labelImg
- images are the original images

2. `cd yolov5`

3. train yolo model (edit data.yaml file to set path to training and validation set)
`python train.py --img 640 --batch 4 --epochs 100 --data ./data.yaml --cfg yolov5s.yaml --weights yolov5s.pt`
- tune hyperparameters accordingly

4. evaluate yolo model against validation set 
- `python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img 640`
- note: run `ls runs/train` to see your experiments. change the above file path to which experiment you want to evaluate

5. check evaluation results
- run `cd runs/val/exp3` or whicever experiment youre currently on
- `open confusion_matrix.png`

5. model can now be used. eg of usage:
```python
import torch
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# Inference
img_path = 'path_to_image.jpg'
results = model(img_path)

# Results
results.print()  # Print results to console
results.save()  # Save results to runs/detect/exp
results.show()  # Display results

# Plot results
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

### Misc.
1. check_font function in yolov5 -> utils -> general.py is edited to disable SSL verification
```python
import ssl
import certifi
from urllib.request import urlopen

def check_font(font='Arial.ttf', progress=True):
    import torch
    from pathlib import Path
    
    # Use certifi's SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    file = Path(font)
    url = 'https://ultralytics.com/assets/' + font
    
    # Download the file with the SSL context
    with urlopen(url, context=ssl_context) as response:
        with open(file, 'wb') as out_file:
            out_file.write(response.read())
```

2. augment_images helps to augment images to generate more, more data for training