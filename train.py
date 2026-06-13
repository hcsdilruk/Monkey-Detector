import tensorflow as tf
from tensorflow.keras import layers, models

# Load Dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128, 128),
    batch_size=32
)

print("Classes:", dataset.class_names)

# Normalize
normalization_layer = layers.Rescaling(1./255)

dataset = dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Model
model = models.Sequential([
    layers.Input(shape=(128,128,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(dataset, epochs=10)

# Save
model.save("monkey_model.keras")

print("✅ Model Saved Successfully")