import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Vocabulary import Vocabulary
from autoencoder import Autoencoder as AE
from WGANGP import WGANGP
import predictor as predictor

import pandas as pd

# Important Path Locations
main_path = "/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/"
# main_path = 'C:\\Users\\Audrey\\eif4e-inhibitor-discovery\\src\\'
model_path = main_path + 'models/'
# model_path = main_path + 'models\\'

dataset_path = main_path + "datasets/subset_100k_LV.csv"
# dataset_path = main_path + "datasets/augmented_dataset_LV.csv"
# dataset_path = main_path + "datasets\\subset_100k_LV.csv"
# dataset_path = main_path + "datasets\\augmented_dataset_LV.csv"

# dataset_path = main_path + 'Data/chembl_roulette.csv'
vocab_path = main_path + 'datasets/subset_500k.csv'
# vocab_path = main_path + 'datasets\\subset_500k.csv'

# Create Vocab
df = pd.read_csv(vocab_path)
selfies = list(df['SELFIES'])
vocab = Vocabulary(selfies)

# Load AE
latent_dim = 256
embedding_dim = 256
lstm_units = 512
epochs = 100
batch_size = 128
batch_norm = True
batch_norm_momentum = 0.9
numb_dec_layer = 2
noise_std = 0.1
input_shape = (vocab.max_len, vocab.vocab_size)
output_dim = vocab.vocab_size

auto = AE(main_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
auto.load_autoencoder_model(model_path + 'AE_model.weights.h5')

# Latent vectors for training data
dataset_df = pd.read_csv(dataset_path)
'''train_selfies = list(dataset_df['SELFIES'])
tok = vocab.tokenize(train_selfies)
encoded = vocab.encode(tok)
latent_vectors = auto.sm_to_lat_model.predict(encoded)
# Convert latent vectors to list of strings
latent_vectors_str = [', '.join(map(str, vec)) for vec in latent_vectors]
# Add the "LV" column to the original DataFrame
dataset_df['LV'] = latent_vectors_str
dataset_df.to_csv("subset_100k_LV.csv", index=False)'''
x_train = predictor.process_lv(dataset_df)

# Create GAN
input_dim = latent_dim
critic_layers_units = [256,256,256]
critic_lr = 0.0001
gp_weight = 10
z_dim  = 64
generator_layers_units = [128,256,256,256,256]
generator_batch_norm_momentum = 0.9
generator_lr = 0.0001
n_epochs = 10001
batch_size = 64
critic_optimizer = 'adam'
generator_optimizer = 'adam'
critic_dropout = 0.2
generator_dropout = 0.2
n_stag_iters = 50
print_every_n_epochs = 250
# run_folder = main_path + "WGANGP\\"
run_folder = main_path + "WGANGP/"
critic_path = model_path + 'critic.keras'
gen_path = model_path + 'generator.keras'
train_distribution = None

gan = WGANGP(main_path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, generator_optimizer, n_stag_iters)
# gan.load_weights(critic_path, gen_path, vocab, auto)

print("Training started...")
gan.train(x_train, batch_size, n_epochs, run_folder, auto, vocab, print_every_n_epochs, train_distribution)
print("Training complete!")