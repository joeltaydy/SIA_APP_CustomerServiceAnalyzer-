# SIA_APP_CustomerServiceAnalyzer-
This is for demo submission for Singapore Airlines AppChallenge 2018. Team AKOV

# Sample Demo
This application utilizes Tensorflow by Google to analyze facial features by customers to deliver insight data as a KPI.\
Classifier is trained with a training set of 30,000 facial expression images.

# Running the Demo
The demo runs on python and utilizes the following dependencies. \
Make sure your tensorflow, opencv-python packages are installed. \
If not, run the commands : \
pip install tensorflow \  
pip install opencv-python

## Open CMD/Terminal and CD to git directory.
Run the command: python cs_analyzer.py

Press Esc to escape the app

# Retraining Tensorflow Image Classifier
python retrain.py --output_graph=retrained_graph.pb --output_labels=labels.txt --architecture=MobileNet_1.0_224 --image_dir=images


