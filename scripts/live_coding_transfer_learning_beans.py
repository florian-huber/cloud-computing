#%% Initialize Tensorflow, check versions and import ResNet50
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
print('TensorFlow version:',tf.__version__)
print('Keras version:',keras.__version__)

image_x, image_y = 224, 224

base_model = keras.applications.resnet.ResNet50(weights='imagenet',
                                                include_top=False,
                                                input_shape=(image_x, image_y, 3))
print(base_model.summary())


#%% Import dataset and process images
def resize_preprocess(image, width, height):
    """Resize images to width x height.
    """
    image = tf.image.resize(image, (width, height)) / 255.0
    return image


image_train, label_train = tfds.as_numpy(tfds.load(
    'beans',
    split= "train", #'test',
    batch_size=-1,
    as_supervised=True,
))

image_val, label_val = tfds.as_numpy(tfds.load(
    'beans',
    split= "validation",
    batch_size=-1,
    as_supervised=True,
))

image_train = resize_preprocess(image_train, image_x, image_y)
image_val = resize_preprocess(image_val, image_x, image_y)
print(f"Imported data. Training data has shape {image_train.shape}.")
print(f"Validation data has shape {image_val.shape}.")


#%% Translate labels to one hot encoding
import pandas as pd
label_train_onehot = pd.get_dummies(label_train).values
label_val_onehot = pd.get_dummies(label_val).values


#%% Compile new model
from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense


# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create prediction layer for classification of our images
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(len(set(label_train)), activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=prediction_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Now print the full model, which will include the layers of the base model plus the dense layer we added
print(model.summary())


#%%
# Train the model
num_epochs = 20

history = model.fit(
    image_train, label_train_onehot,
    validation_data = (image_val, label_val_onehot),
    batch_size=32,
    epochs=num_epochs)
