import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from datetime import datetime

from google.cloud import storage

storage_client = storage.Client()
bucket_name = 'breed_images'

train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.27,
    height_shift_range=0.23,
    shear_range=0.22,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_dir = f'gs://{bucket_name}/train/'  # Path to the training data in GCS

# Create an ImageDataGenerator for loading data
train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# MobileNetV2 as the base model for transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.85)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint callback
checkpoint_path = f"gs://{bucket_name}/project/model/best_model_weights.weights.h5"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Initial training
initial_epochs = 10
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    callbacks=[checkpoint_callback]
)

# Fine-tuning
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(checkpoint_path)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[checkpoint_callback]
)

# Save the fine-tuned model
model_save_path = f"gs://{bucket_name}/project/model/my_trained_model"
model.save(model_save_path)
