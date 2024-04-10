import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define Data Directories
train_data_dir = 'train'
validation_data_dir = 'valid'
test_data_dir = 'test'

# Define Hyperparameters
input_shape = (224, 224, 3)
batch_size = 32  # Adjusted batch size
epochs = 20  # Increased epochs
learning_rate = 1e-4

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Feature Extraction using ResNet50 for Eyes
shared_base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Eyes branch
eyes_head_model = GlobalAveragePooling2D()(shared_base_model.output)
eyes_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(eyes_head_model)
eyes_features = BatchNormalization()(eyes_features)
eyes_features = Dropout(0.5)(eyes_features)
eyes_predictions = Dense(1, activation='sigmoid', name='eyes')(eyes_features)

# Head branch
head_head_model = GlobalAveragePooling2D()(shared_base_model.output)
head_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(head_head_model)
head_features = BatchNormalization()(head_features)
head_features = Dropout(0.5)(head_features)
head_predictions = Dense(1, activation='sigmoid', name='head')(head_features)

# Tongue branch
tongue_head_model = GlobalAveragePooling2D()(shared_base_model.output)
tongue_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(tongue_head_model)
tongue_features = BatchNormalization()(tongue_features)
tongue_features = Dropout(0.5)(tongue_features)
tongue_predictions = Dense(1, activation='sigmoid', name='tongue')(tongue_features)

# Pit branch
pit_head_model = GlobalAveragePooling2D()(shared_base_model.output)
pit_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(pit_head_model)
pit_features = BatchNormalization()(pit_features)
pit_features = Dropout(0.5)(pit_features)
pit_predictions = Dense(1, activation='sigmoid', name='pit')(pit_features)

# Fang branch
fang_head_model = GlobalAveragePooling2D()(shared_base_model.output)
fang_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(fang_head_model)
fang_features = BatchNormalization()(fang_features)
fang_features = Dropout(0.5)(fang_features)
fang_predictions = Dense(1, activation='sigmoid', name='fang')(fang_features)

# Concatenate the features from different branches
concatenated_features = Concatenate()([eyes_features, head_features, tongue_features, pit_features, fang_features])

# Final classification head
final_predictions = Dense(1, activation='sigmoid', name='final_predictions')(concatenated_features)

# Create a model with multiple outputs
model = Model(inputs=shared_base_model.input, outputs=[final_predictions, eyes_predictions, head_predictions, tongue_predictions, pit_predictions, fang_predictions])

# Fine-tune the model by allowing more layers to be trainable
for layer in shared_base_model.layers[-20:]:
    layer.trainable = True


# Adjust class weights
class_weights = {0: 1.3, 1: 0.7}  # Experiment with different weights

# Compile the Model with Adjusted Class Weights
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss={'final_predictions': 'binary_crossentropy', 'eyes': 'binary_crossentropy', 'head': 'binary_crossentropy', 'tongue': 'binary_crossentropy', 'pit': 'binary_crossentropy', 'fang': 'binary_crossentropy'}, metrics=['accuracy'])

# Learning Rate Scheduler
def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 5:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-3
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# # ModelCheckpoint to save the best model
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_final_predictions_accuracy', mode='max')

# Model Training with Learning Rate Scheduler and ModelCheckpoint
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[lr_scheduler, model_checkpoint],
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

# Evaluate the Model on Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Important for confusion matrix and classification report
)

test_loss_and_metrics = model.evaluate(test_generator)
metric_names = model.metrics_names

for metric_name, metric_value in zip(metric_names, test_loss_and_metrics):
    print(f'Test {metric_name}: {metric_value}')



# Confusion Matrix and Classification Report for Test Data
y_true = test_generator.classes
y_pred_final, y_pred_eyes, y_pred_head, y_pred_tongue, y_pred_pit, y_pred_fang = model.predict(test_generator)

# Convert final predictions to binary values
y_pred_final = (y_pred_final > 0.5).astype(int)

# Convert individual predictions to binary values
y_pred_eyes = (y_pred_eyes > 0.5).astype(int)
y_pred_head = (y_pred_head > 0.5).astype(int)
y_pred_tongue = (y_pred_tongue > 0.5).astype(int)
y_pred_pit = (y_pred_pit > 0.5).astype(int)
y_pred_fang = (y_pred_fang > 0.5).astype(int)

# Compute confusion matrices and classification reports for each feature
conf_mat_final = confusion_matrix(y_true, y_pred_final)
class_report_final = classification_report(y_true, y_pred_final, target_names=[str(i) for i in range(2)])

conf_mat_eyes = confusion_matrix(y_true, y_pred_eyes)
class_report_eyes = classification_report(y_true, y_pred_eyes, target_names=[str(i) for i in range(2)])

conf_mat_head = confusion_matrix(y_true, y_pred_head)
class_report_head = classification_report(y_true, y_pred_head, target_names=[str(i) for i in range(2)])

conf_mat_tongue = confusion_matrix(y_true, y_pred_tongue)
class_report_tongue = classification_report(y_true, y_pred_tongue, target_names=[str(i) for i in range(2)])

conf_mat_pit = confusion_matrix(y_true, y_pred_pit)
class_report_pit = classification_report(y_true, y_pred_pit, target_names=[str(i) for i in range(2)])

conf_mat_fang = confusion_matrix(y_true, y_pred_fang)
class_report_fang = classification_report(y_true, y_pred_fang, target_names=[str(i) for i in range(2)])

print("Confusion Matrix for Final Predictions:")
print(conf_mat_final)

print("\nClassification Report for Final Predictions:")
print(class_report_final)

print("\nConfusion Matrix for Eyes Predictions:")
print(conf_mat_eyes)
print("\nClassification Report for Eyes Predictions:")
print(class_report_eyes)

print("\nConfusion Matrix for Head Predictions:")
print(conf_mat_head)
print("\nClassification Report for Head Predictions:")
print(class_report_head)

print("\nConfusion Matrix for Tongue Predictions:")
print(conf_mat_tongue)
print("\nClassification Report for Tongue Predictions:")
print(class_report_tongue)

print("\nConfusion Matrix for Pit Predictions:")
print(conf_mat_pit)
print("\nClassification Report for Pit Predictions:")
print(class_report_pit)

print("\nConfusion Matrix for Fang Predictions:")
print(conf_mat_fang)
print("\nClassification Report for Fang Predictions:")
print(class_report_fang)

# Define function to plot confusion matrix
def plot_confusion_matrix(conf_mat, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Plot confusion matrix for eyes predictions
plot_confusion_matrix(conf_mat_eyes, ['Class 0', 'Class 1'], 'Confusion Matrix for Eyes Predictions')

# Plot confusion matrix for head predictions
plot_confusion_matrix(conf_mat_head, ['Class 0', 'Class 1'], 'Confusion Matrix for Head Predictions')

# Plot confusion matrix for tongue predictions
plot_confusion_matrix(conf_mat_tongue, ['Class 0', 'Class 1'], 'Confusion Matrix for Tongue Predictions')

# Plot confusion matrix for pit predictions
plot_confusion_matrix(conf_mat_pit, ['Class 0', 'Class 1'], 'Confusion Matrix for Pit Predictions')

# Plot confusion matrix for fang predictions
plot_confusion_matrix(conf_mat_fang, ['Class 0', 'Class 1'], 'Confusion Matrix for Fang Predictions')

# Combine individual predictions into a single binary prediction
y_pred_combined = (y_pred_eyes + y_pred_head + y_pred_tongue + y_pred_pit + y_pred_fang) >= 3

# Compute the confusion matrix for the combined prediction
conf_mat_combined = confusion_matrix(y_true, y_pred_combined)

# Plot confusion matrix for combined predictions
plot_confusion_matrix(conf_mat_combined, ['Class 0', 'Class 1'], 'Confusion Matrix for Combined Predictions')