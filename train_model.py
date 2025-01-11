import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from PIL import Image
import gc

def load_images_in_batches(directory, label, batch_size=1000):
    images = []
    labels = []
    count = 0
    
    for filename in os.listdir(directory):
        if count >= batch_size:
            yield np.array(images), np.array(labels)
            images = []
            labels = []
            count = 0
            gc.collect()
            
        img_path = os.path.join(directory, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            images.append(np.array(img))
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            
    if images:
        yield np.array(images), np.array(labels)

def create_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, base_model

def main():
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
    except:
        print("No GPU available or error setting memory growth")

    cancer_dir = "datset/augmented/Cancer"
    non_cancer_dir = "datset/augmented/Non-Cancer"
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        'datset/augmented',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'datset/augmented',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )
    
    model, base_model = create_model()
    
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Initial training with frozen layers...")
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Fine-tuning the model...")
    for layer in base_model.layers:
        layer.trainable = True
        
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    model.save('final_breast_cancer_model.keras')
    print("Training completed. Final model saved as 'final_breast_cancer_model.keras'")

if __name__ == "__main__":
    main()