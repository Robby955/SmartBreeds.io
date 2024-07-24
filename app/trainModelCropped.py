import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import json
from datetime import datetime
from contextlib import redirect_stdout
import sklearn
import openpyxl
# Set up TensorBoard logs directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

breed_dirs = os.listdir(r'C:/Users/robby/OneDrive/Desktop/SmartBreed/SmartDogBreed/DogBreed/Images/images/CroppedImages/train')
def extract_breed(breed_name):
    return breed_name.replace('_', ' ').title()

extracted_breeds = [extract_breed(name) for name in breed_dirs]


results_dir = f'C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\result_summary\\{timestamp}'
os.makedirs(results_dir, exist_ok=True)

EPOCH_NUM_DECREASE=10 # When we start the learning rate decrease
ROTATION_RANGE=45
WIDTH_SHIFT_RANGE=0.30
HEIGHT_SHIFT_RANGE=0.30
SHEAR_RANGE=0.23
ZOOM_RANGE=0.25
BATCH_SIZE=32
DENSE_NODES=2048
DROPOUT_LEVEL=0.925
INIT_LEARNING_RATE=1e-4
INITIAL_EPOCHS=40
FINE_TUNE_EPOCHS=40
FINE_TUNE_LAYERS=1


class MetricsHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'learning_rate': []}

    def on_epoch_end(self, epoch, logs={}):
        self.metrics['loss'].append(logs.get('loss'))
        self.metrics['accuracy'].append(logs.get('accuracy'))
        self.metrics['val_loss'].append(logs.get('val_loss'))
        self.metrics['val_accuracy'].append(logs.get('val_accuracy'))
        self.metrics['learning_rate'].append(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

# Initialize the callback
metrics_history = MetricsHistory()


# Define base directories
cropped_base_dir = Path('C:/Users/robby/OneDrive/Desktop/SmartBreed/SmartDogBreed/DogBreed/Images/images/CroppedImages')
train_dir = cropped_base_dir / 'train'
test_dir = cropped_base_dir / 'test'

# Image Data Generators with enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    shear_range=SHEAR_RANGE,
    zoom_range=ZOOM_RANGE,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# simulated annealing lr schedcule based on metrics or at a minimum cyclic
# Fix any errors or problems in code nad basic webpage for now
# make sure database is working and login/registration
# add instructions and example to front page
# add second and third prediction and confidence


# Learning rate scheduler
def lr_schedule(epoch, lr):
    min_lr = 1e-6  # Set your minimum learning rate
    if epoch < EPOCH_NUM_DECREASE:
        return max(float(lr), min_lr)
    else:
        return max(float(lr * tf.math.exp(-0.1)), min_lr)
lr_scheduler = LearningRateScheduler(lr_schedule)


# MobileNetV2 as the base model for transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Start by freezing base layers

# Custom layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(DENSE_NODES, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_LEVEL)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(INIT_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])


# Define a checkpoint callback to save the best weights
checkpoint_path = "models/best_model_weights.weights.h5"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      monitor='val_accuracy',
                                      mode='max', save_best_only=True,
                                      verbose=1)

# Initial training
initial_epochs = INITIAL_EPOCHS
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=test_generator,
    callbacks=[checkpoint_callback, lr_scheduler, metrics_history]
)

# Fine-tuning
base_model.trainable = True
fine_tune_at = FINE_TUNE_LAYERS
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a low learning rate
model.compile(optimizer=Adam(INIT_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the best weights for fine-tuning
model.load_weights(checkpoint_path)

fine_tune_epochs = FINE_TUNE_EPOCHS
total_epochs = initial_epochs + fine_tune_epochs

# Continue training (fine-tuning)
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=test_generator,
    callbacks=[checkpoint_callback, lr_scheduler,metrics_history]
)


import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


# Predictions and classification report
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
print(confusion_matrix(true_classes, predicted_classes))

# Plotting training information
plot_training_history(history)


from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime
class_labels = list(test_generator.class_indices.keys())
#updated_class_labels = [label_mapping.get(label, label) for label in class_labels]
# Assuming true_classes and predicted_classes are defined
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()
report_df = report_df.rename(columns={'index': 'Breed'})

# Function to extract and clean breed names

# Define training configuration
training_config = {
    "EPOCH_NUM_DECREASE": EPOCH_NUM_DECREASE,
    "ROTATION_RANGE": ROTATION_RANGE,
    "WIDTH_SHIFT_RANGE": WIDTH_SHIFT_RANGE,
    "HEIGHT_SHIFT_RANGE": HEIGHT_SHIFT_RANGE,
    "SHEAR_RANGE": SHEAR_RANGE,
    "ZOOM_RANGE": ZOOM_RANGE,
    "BATCH_SIZE": BATCH_SIZE,
    "DENSE_NODES": DENSE_NODES,
    "DROPOUT_LEVEL": DROPOUT_LEVEL,
    "INIT_LEARNING_RATE": INIT_LEARNING_RATE,
    "INITIAL_EPOCHS": INITIAL_EPOCHS,
    "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS,
    "FINE_TUNE_LAYERS": FINE_TUNE_LAYERS
}

config_df = pd.DataFrame([training_config])

# Combine configuration and results
combined_df = pd.concat([config_df.T, report_df], axis=0)
report_df.set_index('Breed', inplace=True)

# Remove the rows that are not specific to breeds
breed_data_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
breed_data_df = breed_data_df.reset_index().rename(columns={'index': 'Breed'})
breed_data_df = breed_data_df.round(2)


# Save to file as .txt
filename = f"C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\training_history\\training_summary_{timestamp}.txt"
with open(filename, 'w') as f:
    combined_df.to_string(f)

filename2 = f"C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\result_summary\\result_summary_{timestamp}.txt"
with open(filename2, 'w') as f:
    breed_data_df.to_string(f)

print(f"Training summary and results saved to respective directories.")

print(f"Training summary and results saved to {filename}")

import json

# Create a directory to store the confusion matrices
confusion_matrices_dir = Path('confusion_matrices')
confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to hold confusion matrices
breed_confusion_matrices = {}

for breed_idx, breed_name in enumerate(class_labels):
    # Filter out predictions and true labels for this breed
    breed_predictions = [pred == breed_idx for pred in predicted_classes]
    breed_true_labels = [true == breed_idx for true in true_classes]

    # Generate the confusion matrix for this breed
    breed_conf_matrix = confusion_matrix(breed_true_labels, breed_predictions)

    # Add the matrix to our dictionary
    breed_confusion_matrices[
        breed_name] = breed_conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
import json
# Save the confusion matrices to a JSON file
with open(confusion_matrices_dir / f'breed_confusion_matrices_{timestamp}.json', 'w') as f:
    json.dump(breed_confusion_matrices, f, indent=4)

print("Saved breed-specific confusion matrices.")






'''
import matplotlib.pyplot as plt

# List of breeds to analyze
breeds_to_analyze = ['Pug', 'Pekinese', 'Shih']

# Loop through each breed
for breed_to_analyze in breeds_to_analyze:
    breed_index = class_labels.index(breed_to_analyze)  # Get index of the breed

    # Get all indices of this breed in the test set
    breed_indices = [i for i, label in enumerate(true_classes) if label == breed_index]

    # Randomly select a subset of indices for visualization
    sample_indices = random.sample(breed_indices, min(8, len(breed_indices)))

    # Plotting the sample images with true and predicted labels
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Predictions for {breed_to_analyze}", fontsize=14)
    for i, idx in enumerate(sample_indices):
        # Get the image and true label
        x_batch, y_batch = test_generator[idx // BATCH_SIZE]
        image = x_batch[idx % BATCH_SIZE]
        true_label = class_labels[np.argmax(y_batch[idx % BATCH_SIZE])]

        # Get the predicted label
        predicted_label = class_labels[predicted_classes[idx]]
        
        # Plot
        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

'''

current_session_weights_path = '/app/models/best_model_weights.weights.h5'
global_best_weights_path = 'C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\global_best_weights.weights.h5'
global_best_accuracy_path = 'C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\global_best_accuracy.txt'

# Load global best weights if they exist
if os.path.exists(global_best_weights_path):
    model.load_weights(global_best_weights_path)
    with open(global_best_accuracy_path, 'r') as f:
        global_best_accuracy = float(f.read())
else:
    global_best_accuracy = 0.0



# Plotting
def plot_metrics(metrics_history):
    epochs = range(1, len(metrics_history.metrics['accuracy']) + 1)

    plt.figure(figsize=(10, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history.metrics['accuracy'], label='Training Accuracy')
    plt.plot(epochs, metrics_history.metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history.metrics['loss'], label='Training Loss')
    plt.plot(epochs, metrics_history.metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history.metrics['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Call the function to plot metrics
plot_metrics(metrics_history)


import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def plot_metrics(metrics_history):
    epochs = range(1, len(metrics_history.metrics['accuracy']) + 1)
    plt.figure(figsize=(10, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history.metrics['accuracy'], label='Training Accuracy')
    plt.plot(epochs, metrics_history.metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history.metrics['loss'], label='Training Loss')
    plt.plot(epochs, metrics_history.metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history.metrics['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the plot
    plot_filename = f'training_plots_{timestamp}.png'
    plt.savefig(plot_filename)
    print(f"Training plots saved to {plot_filename}")

# Call the function to plot metrics
plot_metrics(metrics_history)


history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_filename = f'training_history_{timestamp}.csv'
history_df.to_csv(history_filename, index=False)
print(f"Training history saved to {history_filename}")

metrics_df = pd.DataFrame(metrics_history.metrics)
metrics_filename = f'custom_metrics_history_{timestamp}.csv'
metrics_df.to_csv(metrics_filename, index=False)
print(f"Custom metrics history saved to {metrics_filename}")


# After completing both initial training and fine-tuning
# Combine histories
def combine_histories(history1, history2):
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    return combined_history

combined_history = combine_histories(history, history_fine)

# Convert the combined history to a DataFrame
combined_metrics_df = pd.DataFrame(combined_history)

# Now you can save the combined metrics DataFrame
combined_metrics_filename = os.path.join(results_dir, f'combined_metrics_history_{timestamp}.csv')
combined_metrics_df.to_csv(combined_metrics_filename, index=False)
print(f"Combined metrics history saved to {combined_metrics_filename}")
combined_metrics_df.to_csv(combined_metrics_filename, index=False)



# Update the plot function to use combined_history
def plot_combined_training_history(combined_history):
    epochs = range(1, len(combined_history['accuracy']) + 1)

    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, combined_history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, combined_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, combined_history['loss'], label='Training Loss')
    plt.plot(epochs, combined_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Now call the updated plotting function
plot_combined_training_history(combined_history)

# Save combined metrics to a CSV for further analysis if needed
combined_metrics_df = pd.DataFrame(combined_history)
combined_metrics_filename = f'combined_metrics_history_{timestamp}.csv'
combined_metrics_df.to_csv(combined_metrics_filename, index=False)

current_best_accuracy = max(max(history.history['val_accuracy']), max(history_fine.history['val_accuracy']))
print(f"Current session's best validation accuracy: {current_best_accuracy}")

if current_best_accuracy > global_best_accuracy:
    print(f"New global best accuracy achieved! Updating from {global_best_accuracy} to {current_best_accuracy}.")
    with open(global_best_accuracy_path, 'w') as f:
        f.write(str(current_best_accuracy))
    os.replace("models/best_model_weights.weights.h5", "global_best_weights.weights.h5")
else:
    print(f"Global best accuracy ({global_best_accuracy}) is higher than current session's best ({current_best_accuracy}).")


from sklearn.metrics import confusion_matrix
import pandas as pd

# Assuming true_labels and predicted_labels are lists of actual and predicted breed labels for your test set
# Also assuming class_labels is a list of all unique breed labels

#conf_matrix = confusion_matrix(true_classes,  predicted_classes, labels=class_labels)

label_mapping = {'n02rob-poodle': 'Poodle', 'n02rob-schnauzer': 'Schnauzer'}
updated_class_labels = [label_mapping.get(label, label) for label in class_labels]
unique_labels = set(true_classes).union(set(predicted_classes))

# Filter the class labels to include only those present in the true and predicted classes
filtered_class_labels = [label for idx, label in enumerate(updated_class_labels) if idx in unique_labels]
conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=[class_labels.index(label) for label in filtered_class_labels])

conf_matrix_df = pd.DataFrame(conf_matrix, index=filtered_class_labels, columns=filtered_class_labels)

# Save the confusion matrix to a CSV file
conf_matrix_filename = os.path.join(results_dir, f'confusion_matrix_{timestamp}.csv')
conf_matrix_df.to_csv(conf_matrix_filename, index=True)

print(f"Confusion matrix saved to {conf_matrix_filename}")


# Initialize a dictionary to store the most common misclassifications
most_common_misclassifications = {}

for breed in class_labels:
    # Ignore the diagonal (true positives)
    incorrect_predictions = conf_matrix_df.loc[breed].drop(breed)
    most_common_misclassification = incorrect_predictions.idxmax()
    most_common_misclassifications[breed] = most_common_misclassification

# Save the most common misclassifications to a file
output_file = "most_common_misclassifications.csv"
most_common_misclassifications_df = pd.DataFrame(list(most_common_misclassifications.items()), columns=['Breed', 'Most Common Misclassification'])
most_common_misclassifications_df.to_csv(output_file, index=False)

print(f"Successfully saved the most common misclassifications to {output_file}")

results_dir = 'C:\\Users\\robby\\OneDrive\\Desktop\\Breed_App\\app\\result_summary'
os.makedirs(results_dir, exist_ok=True)

# Save predictions and true labels
predictions_path = os.path.join(results_dir, f'predictions_{timestamp}.json')
true_labels_path = os.path.join(results_dir, f'true_labels_{timestamp}.json')

with open(predictions_path, 'w') as f:
    json.dump(predicted_classes.tolist(), f)
print(f"Predictions saved to {predictions_path}")

with open(true_labels_path, 'w') as f:
    json.dump(true_classes.tolist(), f)
print(f"True labels saved to {true_labels_path}")

# Save the classification report
report_df_path = os.path.join(results_dir, f'classification_report_{timestamp}.csv')
report_df.to_csv(report_df_path, index=False)
print(f"Classification report saved to {report_df_path}")

# Save the full confusion matrix
conf_matrix_path = os.path.join(results_dir, f'confusion_matrix_{timestamp}.csv')
conf_matrix_df.to_csv(conf_matrix_path, index=False)
print(f"Full confusion matrix saved to {conf_matrix_path}")

# Save the most common misclassifications
output_file = os.path.join(results_dir, "most_common_misclassifications.csv")
most_common_misclassifications_df.to_csv(output_file, index=False)
print(f"Most common misclassifications saved to {output_file}")

'''
# Assuming you've already got your predictions and true labels
label_mapping = {
    'n02rob-poodle': 'Poodle',
    'n02rob-schnauzer': 'Schnauzer'
}
updated_class_labels = [label_mapping.get(label, label) for label in class_labels]

# Ensure you have the label indices for the updated labels
label_indices = {label: idx for idx, label in enumerate(updated_class_labels)}

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=[label_indices[label] for label in updated_class_labels])

# Create the confusion matrix DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=updated_class_labels, columns=updated_class_labels)
confusion_matrices_dir = Path('confusion_matrices')
confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

# Path for the JSON file
confusion_matrix_file_path = confusion_matrices_dir / f'breed_confusion_matrices_{timestamp}.json'

# Save the confusion matrices to a JSON file
with open(confusion_matrix_file_path, 'w') as f:
    json.dump(breed_confusion_matrices, f, indent=4)

print(f"Saved breed-specific confusion matrices to {confusion_matrix_file_path}")
'''


