import os
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load and preprocess Keeshond images
def load_keeshond_images(directory, target_size=(64, 64)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # Assuming the images are in JPG format
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            img = (img - 127.5) / 127.5  # Normalize the images to [-1, 1]
            images.append(img)
    return np.array(images)

keeshond_images = load_keeshond_images(r'C:/Users/robby/OneDrive/Desktop/SmartBreed/SmartDogBreed/DogBreed/Images/images/CroppedImages/train/Keeshond')

# Define the Generator
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(128 * 16 * 16, activation="relu", input_dim=100))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(layers.Activation("tanh"))
    return model


# Define the Discriminator
def build_discriminator(image_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Instantiate the models
generator = build_generator()
discriminator = build_discriminator(keeshond_images.shape[1:])

# Compile the Discriminator
discriminator.compile(optimizer=optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined model (Generator + Discriminator)
z = layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = models.Model(z, valid)
combined.compile(optimizer=optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Training
def train(epochs, batch_size=128):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, keeshond_images.shape[0], batch_size)
        imgs = keeshond_images[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, valid)

        # Progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

# Train the GAN
train(epochs=10, batch_size=32)






