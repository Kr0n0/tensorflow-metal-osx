import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = y_train[:1000]
y_test = y_test[:1000]

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[:1000].reshape(-1, 28*28)
x_test = x_test[:1000].reshape(-1, 28*28)

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
predictions = model(x_train[:1]).numpy()

model.fit(x_train, y_train, epochs=10)

loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("Accuracy: {:5.2}%".format(100*acc))
