**AI_PROJECT**

Wildlife Classification using Machine Learning Models.

**Wildlife Classification**

This repository contains code to build a machine learning model which
can predict the different wildlife species.

**ALGORITHMS USED**

Model is trained using CNN, SVM and CNN-LSTM algorithms.

**Dataset Preparation**

The dataset should be structured as follows:

/Animals

/cat

/dog

/snake

**Kaggle link to dataset:**
[https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset]{.underline}

**INSTRUCTIONS TO RUN THE CODE**

- Open the wildlife_classification.ipynb file in jupyter notebooks.

- load the dataset into the model which can obtained from the Kaggle (I
  am also uploading dataset in the repo you can download dataset from
  there).

- Once the above steps are performed follow the instructions in
  wildlife_classification.ipynb file and execute all cells.

**Training the Model**

- **Data preprocessing:** Pictures are normalized and shrunk to 128 by
  128 pixels.

- **Data Augmentation:** The training set is subjected to augmentations
  such shearing, zooming, and horizontal flipping.

- **Model Training:** One hundred epochs are used to train the CNN
  model. The CNN model\'s retrieved features are used to train SVM.

- **Model Saving:** The CNN, CNN-LSTM, and SVM models are saved as.h5
  and.pkl files, respectively, following training.

> **Evaluating the Model**
>
> The models are evaluated on the test data using the following steps:

- **CNN Model:** Evaluate accuracy using cnn_model.evaluate().

- **SVM model:** Evaluate accuracy using svm_model.predict() on
  extracted features.

- **CNN-LSTM model:** Evaluate accuracy using cnn_lstm_model.evaluate().

> Confusion matrices and classification reports are generated for each
> model.

**Image Classification**

To classify a new image, use the classify_and_display_image() function
with the image path and the trained models (CNN, SVM, CNN-LSTM).

result =classify_and_display_image(\'path/to/image.jpg\', cnn_model,
svm_model, cnn_lstm_model)

**Predictions and Results**

The predictions from CNN, SVM, and CNN-LSTM models are printed,
including the predicted class and the class probabilities. The results
are displayed along with the image.

**Evaluation Metrics**

1.  **Accuracy**

> Accuracy is the primary metric used to evaluate the models. It is
> computed as the percentage of correct predictions made by the model on
> the test data.

2.  **Confusion Matrix**

> The confusion matrix shows the performance of the classification model
> on the test data. It indicates how many instances of each class were
> correctly or incorrectly classified.

3.  **Classification Report**

The classification report includes precision, recall, and F1-score for
each class. It provides a more detailed view of the model\'s
performance, especially when dealing with imbalanced datasets.

4.  **Precision, Recall, F1-Score**

These metrics are important for evaluating the quality of the
classification:

- **Precision**: The proportion of true positive predictions out of all
  > positive predictions.

- **Recall**: The proportion of true positive predictions out of all
  > actual positives.

- **F1-Score**: The harmonic mean of precision and recall.

> The classification report provides these metrics for each class.

**Conclusion**

This project demonstrates how to build a multi-model image
classification pipeline using CNN, SVM, and CNN-LSTM. It leverages
TensorFlow and Keras for deep learning and scikit-learn for machine
learning to deliver accurate and efficient classification. The models
are evaluated using standard metrics like accuracy, confusion matrix,
and classification report.
