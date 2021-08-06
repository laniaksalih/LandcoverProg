import os

import pickle
import joblib
import numpy as np

import sklearn.base
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow.keras as keras

from ModelSessionAbstract import ModelSession
from joblib import dump, load

class KerasDenseFineTune():
    
    

    def __init__(self, underlying_model_file_path, augment_model_file_path, augment_x_train_file_path, augment_y_train_file_path):
        
        AUGMENT_MODEL = load(augment_model_file_path)
        tmodel = keras.models.load_model(underlying_model_file_path, compile=False, custom_objects={
            "jaccard_loss":keras.metrics.mean_squared_error, 
            "loss":keras.metrics.mean_squared_error
        })
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=[tmodel.outputs[0], tmodel.layers[-2].output])
        self.model.compile("sgd","mse")

        self.output_channels = self.model.output_shape[0][3]
        self.output_features = self.model.output_shape[1][3]
        self.input_size = self.model.input_shape[1]

        self.down_weight_padding = 10
        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.augment_x_train = np.load(augment_x_train_file_path)
        self.augment_y_train = np.load(augment_y_train_file_path)
        self.augment_model = sklearn.base.clone(AUGMENT_MODEL)
        self.augment_model = self.augment_model.fit(self.augment_x_train,self.augment_y_train)
        self.augment_model_trained = True
        
        self._last_tile = None
     
    @property
    def last_tile(self):
        return self._last_tile

    def run(self, tile, inference_mode=False):
        if tile.shape[2] == 3: # If we get a 3 channel image, then pretend it is 4 channel by duplicating the first band
            tile = np.concatenate([
                tile,
                tile[:,:,0][:,:,np.newaxis]
            ], axis=2)

        tile = tile / 255.0
        output, output_features = self.run_model_on_tile(tile)

        
        if self.augment_model_trained:
            original_shape = output.shape
            output = output_features.reshape(-1, output_features.shape[2])
            output = self.augment_model.predict_proba(output)
            output = output.reshape(original_shape[0], original_shape[1], -1)

        if not inference_mode:
            self._last_tile = output_features

        return output

    def run_model_on_tile(self, tile, batch_size=32):
        height = tile.shape[0]
        width = tile.shape[1]
        
        output = np.zeros((height, width, self.output_channels), dtype=np.float32)
        output_features = np.zeros((height, width, self.output_features), dtype=np.float32)

        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[self.down_weight_padding:self.down_weight_padding+self.stride_y,
               self.down_weight_padding:self.down_weight_padding+self.stride_x] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size,]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size,]):
                img = tile[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

                batch.append(img)
                batch_indices.append((y_index, x_index))
                batch_count+=1

        model_output = self.model.predict(np.array(batch), batch_size=batch_size, verbose=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[0][i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_output[1][i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output_features = output_features / counts[..., np.newaxis]

        return output, output_features

    def save_state_to(self, directory):
        
        np.save(os.path.join(directory, "augment_x_train.npy"), np.array(self.augment_x_train))
        np.save(os.path.join(directory, "augment_y_train.npy"), np.array(self.augment_y_train))

        joblib.dump(self.augment_model, os.path.join(directory, "augment_model.p"))

        if self.augment_model_trained:
            with open(os.path.join(directory, "trained.txt"), "w") as f:
                f.write("")

        return {
            "message": "Saved model state", 
            "success": True
        }
