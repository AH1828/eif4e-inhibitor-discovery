from keras_tuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from Vocabulary import Vocabulary
import selfies as sf
from rdkit.Chem import MolFromSmiles

from autoencoder import Autoencoder

class AutoencoderHyperModel(HyperModel):
    def __init__(self, model_path, vocab_size, max_len, input_shape, output_dim):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self, hp):
        latent_dim = hp.Choice('latent_dim', values=[128, 256, 512, 1024])
        embedding_dim = hp.Choice('emb_dim', values=[128, 256, 512])
        lstm_units = hp.Choice('lstm_units', values=[256, 512, 1024])
        batch_norm_momentum = hp.Choice('batch_norm_momentum', values=[0.8, 0.9, 0.99])
        numb_dec_layer = hp.Choice('numb_dec_layer', values=[1, 2, 3])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        noise_std = hp.Choice('noise_std', values=[0.1, 0.2, 0.3])

        autoencoder = Autoencoder(
            model_path=self.model_path,
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            output_dim=self.output_dim,
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum,
            noise_std=noise_std,
            numb_dec_layer=numb_dec_layer,
            emb_dim=embedding_dim,
            vocab_size=self.vocab_size,
            max_len=self.max_len
        )
        
        autoencoder.build_model()
        model = autoencoder.model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
        
        return model

path = '/AE/'
file = '/datasets/subset_500k.csv'

run_logistics = open(path + 'logistics.txt', 'w')

selfies_file = pd.read_csv(file)
selfies = list(selfies_file['SELFIES'])
random.shuffle(selfies)

vocab = Vocabulary(selfies)
n_train = int(0.8 * len(selfies))
selfies_train = selfies[:n_train]
selfies_test = selfies[n_train:]

tok_train = vocab.tokenize(selfies_train)
tok_test = vocab.tokenize(selfies_test)
encode_train = np.array(vocab.encode(tok_train))
encode_test = np.array(vocab.encode(tok_test))
X_train = vocab.one_hot_encoder(selfies_train)
Y_train = vocab.get_target(X_train, 'OHE')

run_logistics.write('Vocab Size: ' + str(vocab.vocab_size))
run_logistics.write('\nMax length: ' + str(vocab.max_len))

input_shape = X_train.shape[1:]
output_dim = X_train.shape[-1]

hypermodel = AutoencoderHyperModel(
    model_path=path,
    vocab_size=vocab.vocab_size,
    max_len=vocab.max_len,
    input_shape=input_shape,
    output_dim=output_dim
)

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=1,
    directory='eif4e',
    project_name='ae_hyperparams'
)

tuner.search([encode_train, X_train], Y_train, 
                epochs=100,
                batch_size=128, 
                validation_split=0.1, 
                callbacks=[
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6), 
                    EarlyStopping(monitor='val_loss', patience=3)
                ])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal hyperparameters are:
latent_dim: {best_hps.get('latent_dim')}
lstm_units: {best_hps.get('lstm_units')}
batch_norm_momentum: {best_hps.get('batch_norm_momentum')}
noise_std: {best_hps.get('noise_std')}
emb_dim: {best_hps.get('emb_dim')}
numb_dec_layer: {best_hps.get('numb_dec_layer')}
learning_rate: {best_hps.get('learning_rate')}
""")