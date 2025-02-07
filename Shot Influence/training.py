import time
import numpy as np
import pandas as pd
import tensorflow as tf
import rally_classifier as rc
import train
import util
from sklearn.metrics import brier_score_loss


timestr = time.strftime("%Y%m%d-%H%M%S")     # formats the current time into a string for saving purposes
dataset = pd.read_csv('new_data/dataset.csv')

test_mask = (dataset['match_id'] == 30) | (dataset['match_id'] == 34)     # creates a mask (a Boolean series) that selects rows where the match_id is either 30 or 34
val_ratio = 0.3
encode_columns = []
shot_predictors = ['is_target_turn', 'aroundhead', 'backhand', 'time_proportion']     # list of features
rally_predictors = ['roundscore_diff', 'continuous_score']     # list of features
target = 'is_target_win'

seq_len = dataset.groupby('rally_id').size().max()     # computes the maximum number of shots (or events) in a rally by grouping the data by rally_id
seq_len += 1 if seq_len % 2 == 1 else 2

encoded = pd.get_dummies(dataset, columns=encode_columns)    # encode_columns is empty, so no columns are encoded by default
# if data = {
#     'Color': ['Red', 'Green', 'Blue', 'Red'],
#     'Price': [10, 20, 15, 10]
# }; then
# encoded = Price	Blue	Green	Red	
#             10	  0	    0	    1	
#             20	  0	    1	    0	
#             15	  1	    0	    0	
#             10	  0	    0	    1

codes_type, uniques_type = pd.factorize(encoded['type'])    # encode the type column into numeric labels, storing the unique values in uniques_type
encoded['type'] = codes_type + 1    # add 1 to all indexes to reserve code 0 for paddings
# codes, uniques = pd.factorize(np.array(['b', 'b', 'a', 'c', 'b'], dtype="O"))
#>>> codes
#array([0, 0, 1, 2, 0])
#>>> uniques
#array(['b', 'a', 'c'], dtype=object)

shot_predictors = [c for c in encoded.columns if any(c.startswith(f'{p}_')for p in shot_predictors) or c in shot_predictors]     # filters columns in the encoded dataset, selecting those that are either explicitly in shot_predictors or start with any of the shot predictors.
train_data, val_data, test_data = train.split_data(encoded, val_ratio=val_ratio, test_mask=test_mask)

(train_shots, train_shot_types), (train_rallies, train_target, train_rally_id) = train.prepare_data(train_data, shot_attributes = [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], rally_attributes = [rally_predictors, target, 'rally_id'], pad_to=seq_len)
# ensures that sequences (e.g., rallies) are padded to the specified length (seq_len)

(val_shots, val_shot_types), (val_rallies, val_target, val_rally_id) = train.prepare_data(val_data, [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], [rally_predictors, target, 'rally_id'], pad_to=seq_len)
seq_len = train_shots.shape[1]    # match the number of time steps (shots) in the training data

train_hit_area_encoded = train_shot_types[:, :, 0].copy()
train_player_area_encoded = train_shot_types[:, :, 1].copy()
train_opponent_area_encoded = train_shot_types[:, :, 2].copy()
train_shot_types = train_shot_types[:, :, 3].copy()
train_time_proportion = train_shots[:, :, 2].copy()             # extracts 3rd feature (index 2) from train_shots as time proportion
train_shots = np.delete(train_shots, 2, axis=2)    # deletes 3rd feature (time proportion) from the train_shots data

val_hit_area_encoded = val_shot_types[:, :, 0].copy()
val_player_area_encoded = val_shot_types[:, :, 1].copy()
val_opponent_area_encoded = val_shot_types[:, :, 2].copy()
val_shot_types = val_shot_types[:, :, 3].copy()
val_time_proportion = val_shots[:, :, 2].copy()             # time proportion
val_shots = np.delete(val_shots, 2, axis=2)

shot_predictors.remove('time_proportion')    # avoid using time_proportion as a shot predictor in the model 


regularizer = tf.keras.regularizers.l2(0.01)    # apply L2 regularizer with a value of 0.01
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']     # performance metric for training
epochs = 100

callbacks = tf.keras.callbacks.EarlyStopping(min_delta=0.002, patience=15, restore_best_weights=True)    # stops training if validation loss does not improve by at least 0.002 for 15 consecutive epochs. 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1)    # logs training metrics for visualization in TensorBoard

# Model architecture setup
n_shot_types = len(uniques_type) + 1
n_area_types = encoded['player_location_area'].nunique() + 1
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer,
              'activation': 'relu'}    # define convolutional neural network (CNN) layer's configuration, use 32 filters of size 3
rnn_kwargs = {'units': 32, 'kernel_regularizer': regularizer}
dense_kwargs = {'kernel_regularizer': regularizer}

batch_size = 32     # number of samples per batch when training the model


# Avoid tensorflow use full memory by ensuring that TensorFlow only uses as much GPU memory as necessary
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

prediction_model, attention_model = rc.bad_net((seq_len, len(shot_predictors)),
                                               embed_types_size=n_shot_types,
                                               embed_area_size=n_area_types,
                                               rally_info_shape=len(rally_predictors),
                                               cnn_kwargs=cnn_kwargs,
                                               rnn_kwargs=rnn_kwargs,
                                               dense_kwargs=dense_kwargs)
prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train model
train_x = [train_hit_area_encoded, train_player_area_encoded, train_opponent_area_encoded, train_shots, train_shot_types, train_time_proportion, train_rallies]
val_x = [val_hit_area_encoded, val_player_area_encoded, val_opponent_area_encoded, val_shots, val_shot_types, val_time_proportion, val_rallies]

history = prediction_model.fit(train_x, train_target,
                               validation_data=(val_x, val_target),
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=[callbacks])    # returns value for loss, aus & binary accuracy
# Access the history of the model training
# history.history
# output example: {
#     'loss': [0.693, 0.625, 0.592, ...],  # Training loss over epochs
#     'AUC': [0.54, 0.61, 0.68, ...],      # Area Under Curve (AUC) on the training data
#     'binary_accuracy': [0.51, 0.62, 0.69, ...],  # Binary accuracy during training
#     'val_loss': [0.691, 0.610, 0.586, ...],      # Validation loss during training
#     'val_AUC': [0.53, 0.60, 0.70, ...],          # Validation AUC
#     'val_binary_accuracy': [0.50, 0.63, 0.71, ...] # Validation accuracy
# }

# Plot training history
# import matplotlib.pyplot as plt

# # Plot training & validation loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.show()

# # Plot training & validation accuracy
# plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()


# Save model weights
MODEL_NAME = 'SPECIFY_NAME'
model_file = './model/' + MODEL_NAME + '/' + timestr + '/'
prediction_model.save_weights(model_file, save_format='tf')
