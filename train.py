import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import build_emotion_model

# --- DIRECTORIES ---
BASE_DIR = "../emotions2/"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# --- SETTINGS ---
IMG_SIZE = 48
BATCH_SIZE = 32 # Lowered from 64 to save RAM on your PC
EPOCHS = 30     # 30 epochs is a good balance for a solid presentation

# --- DATA PREPARATION ---
# Simplified augmentation to reduce CPU load while keeping 'scipy' happy
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='grayscale',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- MODEL COMPILATION ---
model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    'emotion_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    verbose=1, 
    min_lr=0.00001
)

# --- TRAINING ---
print(f"\n[INFO] Starting training on {train_generator.samples} images...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, reduce_lr]
)