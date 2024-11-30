import tensorflow as tf
from keras import layers

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, model_input_shape=(32, 32, 3)):
        super(SimpleCNN, self).__init__()
        self.model_input_shape = model_input_shape
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=model_input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def build_model(self):
        dummy_input = tf.keras.Input(shape=self.model_input_shape)
        self(dummy_input)
        self.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
