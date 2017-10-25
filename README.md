# Image-Classifier
Convolutional Nueral Network trained with Resnet 50 Model and Imagenet to identify canine breed
Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the Machine Learning Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

Sample Output

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app. Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. Your imperfect solution will nonetheless create a fun user experience!

Project Instructions

Instructions

Clone the repository and navigate to the downloaded folder.

	git clone https://github.com/udacity/dog-project.git
	cd dog-project
Download the dog dataset. Unzip the folder and place it in the repo, at location path/to/dog-project/dogImages.

Download the human dataset. Unzip the folder and place it in the repo, at location path/to/dog-project/lfw. If you are using a Windows machine, you are encouraged to use 7zip to extract the folder.

Donwload the VGG-16 bottleneck features for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

Install the necessary Python packages.

For Mac/OSX:

	conda env create -f requirements/aind-dog-mac.yml
	source activate aind-dog
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
For Windows:

	conda env create -f requirements/aind-dog-windows.yml
	activate aind-dog
	set KERAS_BACKEND=tensorflow 
	python -c "from keras import backend"
Open the notebook and follow the instructions.

	jupyter notebook dog_app.ipynb
NOTE: While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. Unless requested, do not modify code that has already been included.

Amazon Web Services

Instead of training your model on a local CPU (or GPU), you could use Amazon Web Services to launch an EC2 GPU instance. Please refer to the Udacity instructions for setting up a GPU instance for this project.

Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project rubric. Review this rubric thoroughly, and self-evaluate your project before submission. All criteria found in the rubric must meet specifications for you to pass.

Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:

The dog_app.ipynb file with fully functional code, all code cells executed and displaying output, and all questions answered.
An HTML or PDF export of the project notebook with the name report.html or report.pdf.
Any additional images used for the project that were not supplied to you for the project. Please do not include the project data sets in the dogImages/ or lfw/ folders. Likewise, please do not include the bottleneck_features/ folder.
Alternatively, your submission could consist of the GitHub link to your repository.


Project Rubric

Files Submitted

Criteria	Meets Specifications
Submission Files	The submission includes all required files.
Documentation

Criteria	Meets Specifications
Comments	The submission includes comments that describe the functionality of the code.
Step 1: Detect Humans

Criteria	Meets Specifications
Question 1: Assess the Human Face Detector	The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected human face.
Question 2: Assess the Human Face Detector	The submission opines whether Haar cascades for face detection are an appropriate technique for human detection.
Step 2: Detect Dogs

Criteria	Meets Specifications
Question 3: Assess the Dog Detector	The submission returns the percentage of the first 100 images in the dog and human face datasets with a detected dog.
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Criteria	Meets Specifications
Model Architecture	The submission specifies a CNN architecture.
Train the Model	The submission specifies the number of epochs used to train the algorithm.
Test the Model	The trained model attains at least 1% accuracy on the test set.
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

Criteria	Meets Specifications
Obtain Bottleneck Features	The submission downloads the bottleneck features corresponding to one of the Keras pre-trained models (VGG-19, ResNet-50, Inception, or Xception).
Model Architecture	The submission specifies a model architecture.
Question 5: Model Architecture	The submission details why the chosen architecture succeeded in the classification task and why earlier attempts were not as successful.
Compile the Model	The submission compiles the architecture by specifying the loss function and optimizer.
Train the Model	The submission uses model checkpointing to train the model and saves the model with the best validation loss.
Load the Model with the Best Validation Loss	The submission loads the model weights that attained the least validation loss.
Test the Model	Accuracy on the test set is 60% or greater.
Predict Dog Breed with the Model	The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN.
Step 6: Write your Algorithm

Criteria	Meets Specifications
Write your Algorithm	The submission uses the CNN from Step 5 to detect dog breed. The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.
Step 7: Test your Algorithm

Criteria	Meets Specifications
Test Your Algorithm on Sample Images!	The submission tests at least 6 images, including at least two human and two dog images.
Question 6: Test Your Algorithm on Sample Images!	The submission discusses performance of the algorithm and discusses at least three possible points of improvement.
Suggestions to Make your Project Stand Out!

(Presented in no particular order ...)

(1) Augment the Training Data

Augmenting the training and/or validation set might help improve model performance.

(2) Turn your Algorithm into a Web App

Turn your code into a web app using Flask or web.py!

(3) Overlay Dog Ears on Detected Human Heads

Overlay a Snapchat-like filter with dog ears on detected human heads. You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face. If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist here.

(4) Add Functionality for Dog Mutts

Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.

(5) Experiment with Multiple Dog/Human Detectors

Perform a systematic evaluation of various methods for detecting humans and dogs in images. Provide improved methodology for the face_detector and dog_detector functions.
