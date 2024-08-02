import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib
import h5py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Vocabulary import Vocabulary
from autoencoder import Autoencoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class Predictor():
    def __init__(self, path, property, load, split, vocab, autoencoder, df, suffix='', hyperparams=[1,256,0.0001]):
        self.path = path
        self.property = property
        self.dropout = 0.3
        self.n_layers = hyperparams[0]
        self.n_units = hyperparams[1]
        self.learning_rate = hyperparams[2]
        self.n_epochs = 1000
        self.batch_size = 32
        self.validation_split = 0.1
        self.load = load
        self.split = split
        self.vocab = vocab
        self.auto = autoencoder
        self.data = df
        self.input_length = 256

        if not load:
            self.get_latent_representations()
            self.train_test_split()
        self.build_model()
        # self.build_random_forest()

        if self.load:
            self.load_model(self.property, suffix)
    
    # Convert SELFIES to latent vector
    def selfies_to_latentvector(self, selfies):
        tokens = self.vocab.tokenize(selfies)
        encoded = np.array(self.vocab.encode(tokens))
        latent_vectors = self.auto.sm_to_lat_model.predict(encoded)
        return latent_vectors
    
    # Add column for latent representations into self.data dataframe
    def get_latent_representations(self):
        selfies = list(self.data['SELFIES'])
        lat_vecs = self.selfies_to_latentvector(selfies).tolist()
        self.data['LV'] = lat_vecs

    # Create train and test data
    def train_test_split(self):
        # Shuffle dataframe
        self.data = self.data.sample(frac=1, ignore_index=True)

        # Create X and Y train
        lat_vecs = list(self.data['LV'])
        property = list(self.data[self.property])

        self.range = max(property) - min(property)

        train_length = int(len(lat_vecs) * self.split)
        self.X_train = np.array(lat_vecs[:train_length])
        self.Y_train = np.array(property[:train_length])
        self.X_test = np.array(lat_vecs[train_length:])
        self.Y_test = np.array(property[train_length:])

        # Get input length from latent vector
        self.input_length = len(self.X_train[0])

    # Create neural network model
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_length)))
        for _ in range(self.n_layers):
            model.add(Dense(self.n_units, activation='relu'))
            model.add(Dropout(rate=self.dropout))
        model.add(Dense(1, activation='linear'))

        self.model = model
        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss=MSE, optimizer=opt, metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])

    # Create random forest model
    def build_random_forest(self):
        self.rf_model = RandomForestRegressor(n_estimators=1000)

    # Compile and train neural network model
    def train_model(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
        mc = ModelCheckpoint(self.path + 'best_model_' + self.property + '.keras', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        result = self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks = [es, mc], verbose=0)
        
        # Training curve for MSE
        plt.plot(result.history['loss'], label='Train')
        plt.plot(result.history['val_loss'], label='Validation')
        plt.title('Training Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_loss_' + self.property + '.png')
        plt.close()

        # Training curve for RMSE
        plt.plot(result.history['root_mean_squared_error'], label='Train')
        plt.plot(result.history['val_root_mean_squared_error'], label='Validation')
        plt.title('Training RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_rmse_' + self.property + '.png')
        plt.close()

        # Save model
        self.model.save(self.path + 'model_' + self.property + '.keras')
        print('NN Done!')

    # Train random forest model
    def train_random_forest(self):
        self.rf_model.fit(self.X_train, self.Y_train)
        self.save_random_forest()

    # Save random forest model
    def save_random_forest(self):
        joblib.dump(self.rf_model, self.path + 'rf_model_' + self.property + '.joblib')

    # Load random forest model
    def load_random_forest(self):
        self.rf_model = joblib.load(self.path + 'rf_model_' + self.property + '.joblib')

    # Load pre-trained predictor model
    def load_model(self, property, suffix=''):
        # path = "C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\models\\" + property + "_model.weights.h5"
        path = "/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/models/" + property + "_model.weights.h5"
        with h5py.File(path, 'r') as f:
            for layer in self.model.layers:
                if layer.name in f.keys():
                    g = f[layer.name]
                    weights = [g[var] for var in g.keys()]
                    layer.set_weights(weights)
    
    # Evaluate model performance
    def evaluate(self):
        # Evaluate neural network model
        nn_performance = self.model.evaluate(self.X_test, self.Y_test)
        nn_mse = nn_performance[0]
        nn_rmse = nn_performance[1]
        nn_mape = nn_performance[2]

        # Evaluate random forest model
        rf_predictions = self.rf_model.predict(self.X_test)
        rf_mse = mean_squared_error(self.Y_test, rf_predictions)
        rf_rmse = np.sqrt(rf_mse)
        rf_mape = mean_absolute_percentage_error(self.Y_test, rf_predictions)

        # Save results
        results_path = os.path.join(self.path, f'evaluation_{self.property}.txt')
        with open(results_path, 'w', encoding="utf-8") as results:
            results.write('Neural Network:\n')
            results.write(f'MSE: {round(nn_mse, 4)}\nRMSE: {round(nn_rmse, 4)}\nMAPE: {round(nn_mape, 4)}\n')
            results.write('\nRandom Forest:\n')
            results.write(f'MSE: {round(rf_mse, 4)}\nRMSE: {round(rf_rmse, 4)}\nMAPE: {round(rf_mape, 4)}\n')

        return (nn_mse, nn_rmse, nn_mape), (rf_mse, rf_rmse, rf_mape)

    # Make predictions for molecular property
    def predict(self, selfies, string=True):
        if string:
            lat_vecs = self.selfies_to_latentvector(selfies)
            nn_predictions = self.model.predict(lat_vecs)
            # rf_predictions = self.rf_model.predict(lat_vecs)
        else:
            lat_vecs = selfies
            nn_predictions = self.model(lat_vecs)
            # rf_predictions = self.rf_model.predict(lat_vecs)
        
        nn_predictions = [p[0] for p in nn_predictions]
        # rf_predictions = [p for p in rf_predictions]

        if tf.is_tensor(nn_predictions[0]):
            nn_predictions = [p.numpy() for p in nn_predictions]
            
        return nn_predictions

def repurpose_for_target(path, property, vocab, auto, df_train, df_repurpose, save_path):
    # Load predictor
    predictor = Predictor(path, property, True, 0.8, vocab, auto, df_train, suffix='_500k')

    # Make predictions
    all_selfies = list(df_repurpose['SELFIES'])
    #all_selfies = process_lv(df_repurpose)

    print('Predictions starting...')
    predictions = predictor.predict(all_selfies, string=True)
    print('Predictions complete!')

    df_repurpose[property] = predictions
    df_repurpose.to_csv(save_path, index=False)

def process_lv(df):
    lv_strings = list(df['LV'])

    lv_nums = []
    for lv in lv_strings:
        try:
            cur = lv.split(', ')
            for i in range(len(cur)):
                cur[i] = float(cur[i].replace(']', '').replace('[', ''))
            lv_nums.append(np.array(cur, dtype='float32'))
        except ValueError:
            cur = lv.replace(']', '').replace('[', '')
            cur = cur.split()
            for i in range(len(cur)):
                cur[i] = float(cur[i])
            lv_nums.append(np.array(cur, dtype='float32'))

    return np.array(lv_nums)

# Function to train and evaluate models for pIC50
def run_pIC50():
    prefix = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    path = prefix + 'predictor/'
    property = 'pIC50'
    vocab_df = pd.read_csv(prefix + 'datasets/subset_500k.csv')
    ae_path = prefix + 'models/AE_model.weights.h5'
    encoder_path = prefix + 'models/encoder_model.weights.h5'
    decoder_path = prefix + 'models/decoder_model.weights.h5'
    df = pd.read_csv(prefix + 'datasets/augmented_dataset.csv')
    # prefix = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
    # path = prefix + 'predictor\\'
    # property = 'pIC50'
    # vocab_df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')
    # ae_path = prefix + 'models\\AE_model.weights.h5'
    # encoder_path = prefix + 'models\\encoder_model.weights.h5'
    # decoder_path = prefix + 'models\\decoder_model.weights.h5'
    # df = pd.read_csv(prefix + 'datasets\\augmented_dataset.csv')

    vocab = Vocabulary(list(vocab_df['SELFIES']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    auto.load_encoder_model(encoder_path)
    auto.load_decoder_model(decoder_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k')
    predictor.train_model()
    predictor.train_random_forest()
    predictor.evaluate()

# Function to train and evaluate models for LogP
def run_LogP():
    prefix = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    path = prefix + 'predictor/'
    property = 'LogP'
    vocab_df = pd.read_csv(prefix + 'datasets/subset_500k.csv')
    ae_path = prefix + 'models/AE_model.weights.h5'
    encoder_path = prefix + 'models/encoder_model.weights.h5'
    decoder_path = prefix + 'models/decoder_model.weights.h5'
    df = pd.read_csv(prefix + 'datasets/augmented_dataset.csv')
    # prefix = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
    # path = prefix + 'predictor\\'
    # property = 'LogP'
    # vocab_df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')
    # ae_path = prefix + 'models\\AE_model.weights.h5'
    # encoder_path = prefix + 'models\\encoder_model.weights.h5'
    # decoder_path = prefix + 'models\\decoder_model.weights.h5'
    # df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')

    vocab = Vocabulary(list(vocab_df['SELFIES']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    auto.load_encoder_model(encoder_path)
    auto.load_decoder_model(decoder_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.train_random_forest()
    predictor.evaluate()

# Function to train and evaluate models for MW
def run_MW():
    prefix = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    path = prefix + 'predictor/'
    property = 'MW'
    vocab_df = pd.read_csv(prefix + 'datasets/subset_500k.csv')
    ae_path = prefix + 'models/AE_model.weights.h5'
    encoder_path = prefix + 'models/encoder_model.weights.h5'
    decoder_path = prefix + 'models/decoder_model.weights.h5'
    df = pd.read_csv(prefix + 'datasets/augmented_dataset.csv')
    # prefix = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
    # path = prefix + 'predictor\\'
    # property = 'MW'
    # vocab_df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')
    # ae_path = prefix + 'models\\AE_model.weights.h5'
    # encoder_path = prefix + 'models\\encoder_model.weights.h5'
    # decoder_path = prefix + 'models\\decoder_model.weights.h5'
    # df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')

    vocab = Vocabulary(list(vocab_df['SELFIES']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    auto.load_encoder_model(encoder_path)
    auto.load_decoder_model(decoder_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.train_random_forest()
    predictor.evaluate()

# Function to train and evaluate models for SAS
def run_SAS():
    prefix = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    path = prefix + 'predictor/'
    property = 'SAS'
    vocab_df = pd.read_csv(prefix + 'datasets/subset_500k.csv')
    ae_path = prefix + 'models/AE_model.weights.h5'
    encoder_path = prefix + 'models/encoder_model.weights.h5'
    decoder_path = prefix + 'models/decoder_model.weights.h5'
    df = pd.read_csv(prefix + 'datasets/augmented_dataset.csv')
    # prefix = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
    # path = prefix + 'predictor\\'
    # property = 'SAS'
    # vocab_df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')
    # ae_path = prefix + 'models\\AE_model.weights.h5'
    # encoder_path = prefix + 'models\\encoder_model.weights.h5'
    # decoder_path = prefix + 'models\\decoder_model.weights.h5'
    # df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')

    vocab = Vocabulary(list(vocab_df['SELFIES']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    auto.load_encoder_model(encoder_path)
    auto.load_decoder_model(decoder_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.train_random_forest()
    predictor.evaluate()

# Function to train and evaluate models for QED
def run_QED():
    prefix = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    path = prefix + 'predictor/'
    property = 'QED'
    vocab_df = pd.read_csv(prefix + 'datasets/subset_500k.csv')
    ae_path = prefix + 'models/AE_model.weights.h5'
    encoder_path = prefix + 'models/encoder_model.weights.h5'
    decoder_path = prefix + 'models/decoder_model.weights.h5'
    df = pd.read_csv(prefix + 'datasets/augmented_dataset.csv')
    # prefix = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
    # path = prefix + 'predictor\\'
    # property = 'QED'
    # vocab_df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')
    # ae_path = prefix + 'models\\AE_model.weights.h5'
    # encoder_path = prefix + 'models\\encoder_model.weights.h5'
    # decoder_path = prefix + 'models\\decoder_model.weights.h5'
    # df = pd.read_csv(prefix + 'datasets\\subset_500k.csv')

    vocab = Vocabulary(list(vocab_df['SELFIES']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    auto.load_encoder_model(encoder_path)
    auto.load_decoder_model(decoder_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.train_random_forest()
    predictor.evaluate()

if __name__ == '__main__':
    run_pIC50()
    run_LogP()
    run_MW()
    # run_SAS()
    # run_QED()