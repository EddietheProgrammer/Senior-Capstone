# Capstone Project

**Make sure you have [python](https://www.python.org/downloads/) installed.**

# Installation
Right now I am using the [baseballcv](https://github.com/dylandru/BaseballCV/tree/main) package to load pre-trained models and datasets to make our lives easier. 

Here's how you should set this up (I recommend an IDE like VS Code; video files kinda crash Jupyter Notebook).

```bash
git clone https://github.com/EddietheProgrammer/Senior-Capstone.git
```

Please keep in mind Windows file system is different with \ instead of / (Good going Microsoft):
```bash
cd Senior\ Capstone/
```

I recommend creating a virtual environment so you don't run into any issues with package versions.
```bash
python -m venv myenv # myenv = my environment
```

You then activate it with
```bash
source myenv/bin/activate # Note: This may be different for Windows so let me know if this doesn't work
```

Lastly, you will need to:
```bash
pip install -r requirements.txt
```
**NOTE**: I need to update some things in requirements.txt so you may still encounter `ModuleNotFoundError`. Just
`pip install` the corresponding package name for now. I'll try to fix it.

This will install everything you need. I will update if there's additional packages. To deactivate your environment, simply type `deactivate` in the terminal.

Also, please use a .gitignore file for files you don't want merged with the main branch. 
i.e. the `myenv` folder. To do this, create a file called .gitignore then type your environment name in the file. It should be greyed out.

# Running
I've made running this thing pretty simple. You just run things in `main.py`. The only work you will have to do is 
by populating the `assets/` folder with video data, specifically from MLB feed. Some examples of running include:
```python
from baseballcv.functions import LoadTools
from ultralytics import YOLO
from rtmlib import PoseTracker, Body
from threeD_estimation import ThreeDEstimator

tools = LoadTools()
pose_model = YOLO(tools.load_model('phc_detector', model_type='YOLO')) 
tracker = PoseTracker(Body, 7, False, mode='performance', backend='onnxruntime', device='mps')

estimator = ThreeDEstimator(video_path='assets/test.mp4', json_name='test') # The video file in your assets folde + what you want to call the json output

estimator.extract_2D_coordinates(pose_model, tracker) # Exctracts the coordinates and writes it to a json file
estimator.write_2D_output(pose_model, tracker, True) # Writes the 2D pose model
estimator.write_3D_output() # Writes the 3D pose model
predicted_keypoints = estimator.run_model()
```

## Editors Note:
You may also need to install the following:
```bash
pip install git+https://github.com/Jensen-holm/statcast-era-pitches.git 
```

# Contributing
If you want to contribute, fork the repository then use the following commands:
```bash
git checkout -b feature/YourFeature # Can change the naming
git commit -m "Adding files" # This is done afer you mainipulate the repo
git push origin feature/YourFeature # Pushes changes to your master branch
```
After you push, open a pull request and I will review and change if I like it.