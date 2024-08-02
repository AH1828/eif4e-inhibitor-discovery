import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import statistics
from rdkit.Chem import MolFromSmiles, Descriptors
import random
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D as ax
import time

from tensorflow.keras import backend as K

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from WGANGP import WGANGP
from Vocabulary import Vocabulary
from predictor import Predictor
from autoencoder import Autoencoder as AE

class ParetoGA(WGANGP):
    def __init__(self, path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters, predictors, df, suffix=''):
        super().__init__(path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters)
        self.predictors = predictors
        self.find_scale_bounds(df)
        self.replacement_start_epoch = 0
        self.replacement_percent = 0.1
        self.current_samples = []
        self.suffix = suffix

        # Initialize fitness lists
        self.train_fitness = []
        self.gen_fitness_avg = []
        self.gen_fitness_max = []
        self.gen_fitness_min = []
        self.pareto_fronts_over_time = []

    def find_scale_bounds(self, df):
        mw = list(df['MW'])
        logp = list(df['LogP'])
        pIC50 = list(df['pIC50'])

        self.scale_bounds_mw = (min(mw), max(mw))
        self.scale_bounds_logp = (min(logp), max(logp))
        self.scale_bounds_pIC50 = (min(pIC50), max(pIC50))

        fitness_vals = self.calculate_fitness([pIC50, mw, logp], lv=False)
        self.scale_bounds_fitness = (min(fitness_vals), max(fitness_vals))

    def calculate_fitness(self, samples, lv=True):
        if lv:
            samples = np.array(samples)
            pIC50s = self.predictors['pIC50'].predict(samples, string=False)
            MWs = self.predictors['MW'].predict(samples, string=False)
            LogPs = self.predictors['LogP'].predict(samples, string=False)
            pIC50_vals = pIC50s if isinstance(pIC50s, list) else pIC50s.tolist()
            mw_vals = MWs if isinstance(MWs, list) else MWs.tolist()
            logp_vals = LogPs if isinstance(LogPs, list) else LogPs.tolist()
            i = 0
            while i < len(pIC50_vals):
                if mw_vals[i] <= 0 or logp_vals[i] <= 0:
                    pIC50_vals[i] = 0
                    mw_vals[i] = 10000
                    logp_vals[i] = 100
                    i -= 1
                i += 1
        else:
            pIC50_vals = samples[0]
            mw_vals = samples[1]
            logp_vals = samples[2]

        fitness_vals = []
        for i in range(len(pIC50_vals)):
            mw = mw_vals[i]
            logp = logp_vals[i]
            pIC50 = pIC50_vals[i]

            # Scaling logic to ensure correct optimization direction
            scaled_mw = (self.scale_bounds_mw[1] - mw) / (self.scale_bounds_mw[1] - self.scale_bounds_mw[0])
            scaled_logp = (self.scale_bounds_logp[1] - logp) / (self.scale_bounds_logp[1] - self.scale_bounds_logp[0])
            scaled_pIC50 = (pIC50 - self.scale_bounds_pIC50[0]) / (self.scale_bounds_pIC50[1] - self.scale_bounds_pIC50[0])

            fitness_vals.append([scaled_mw, scaled_logp, scaled_pIC50])

        return fitness_vals

    def train_critic(self, x_train):
        data = x_train
        noise = np.random.uniform(-1,1,(self.batch_size, self.z_dim))

        with tf.GradientTape() as critic_tape:
            self.critic.training = True

            generated_data = self.generator(noise)
            self.current_samples.extend(generated_data.numpy().tolist())
            
            real_output = self.critic(data)
            fake_output = self.critic(generated_data)
            
            critic_loss = K.mean(fake_output) - K.mean(real_output)
            self.critic_loss_real.append(K.mean(real_output))
            self.critic_loss_fake.append(K.mean(fake_output))
            
            alpha = tf.random.uniform((self.batch_size,1))
            interpolated_samples = alpha*data +(1-alpha)*generated_data
            
            with tf.GradientTape() as t:
                t.watch(interpolated_samples)
                interpolated_samples_output = self.critic(interpolated_samples)
                
            gradients = t.gradient(interpolated_samples_output, [interpolated_samples])
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(gradients_sqr, axis = np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = K.square(1-gradient_l2_norm)
            gp =  K.mean(gradient_penalty)
            
            self.gradient_penalty_loss.append(gp)
            critic_loss = critic_loss +self.gp_weight*gp
            
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        
        return critic_loss

    def train_generator(self):
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        
        with tf.GradientTape() as generator_tape:
            self.generator.training = True
            generated_data = self.generator(noise)
            self.current_samples.extend(generated_data.numpy().tolist())
            
            fake_output = self.critic(generated_data)
            gen_loss = -K.mean(fake_output)
            
        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        return gen_loss

    def train(self, x_train, batch_size, epochs, run_folder, autoencoder, vocab, print_every_n_epochs, train_distribution, critic_loops=5, replace_every_n_epochs=100):        
        self.n_critic = critic_loops
        self.autoencoder = autoencoder
        self.vocab = vocab
        self.replace_every_n_epochs = replace_every_n_epochs

        self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
        
        train_start = time.time()
        
        self.g_loss_log = []
        self.critic_loss_log =[]
        self.critic_loss = []
        self.g_loss = []
        self.train_fitness = []
        self.gen_fitness_avg = []
        self.gen_fitness_max = []
        self.gen_fitness_min = []
        self.pareto_fronts_over_time = []

        for epoch in range(self.epoch, self.epoch+epochs):
            critic_loss_per_batch = []
            g_loss_per_batch = []
            batches_done = 0
            
            for i, batch in enumerate(self.data):
                loss_d = self.train_critic(batch)
                critic_loss_per_batch.append(loss_d)
                
                if i % self.n_critic == 0:
                    loss_g = self.train_generator()
                    g_loss_per_batch.append(loss_g)
                    batches_done = batches_done +  self.n_critic
                
                if i == len(self.data) -1:
                    self.critic_loss_log.append([time.time()-train_start, epoch, np.mean(critic_loss_per_batch)])
                    self.g_loss_log.append([time.time()-train_start, epoch, np.mean(g_loss_per_batch)])
                    self.critic_loss.append(np.mean(critic_loss_per_batch))
                    self.g_loss.append(np.mean(g_loss_per_batch))		   
                    print( 'Epochs {}: D_loss = {}, G_loss = {}'.format(epoch, self.critic_loss_log[-1][2], self.g_loss_log[-1][2]))

                    if (epoch % print_every_n_epochs) == 0:
                        print('Saving...')
                        self.save_model(run_folder)
                        self.plot_loss(run_folder)
                        weights_dir = os.path.join(run_folder, 'weights')
                        os.makedirs(weights_dir, exist_ok=True)
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights_' + self.suffix + '.keras'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights_' + self.suffix + '.keras'))
                        self.sample_data(200, run_folder, train_distribution, save=True)
                    
                    if (epoch % replace_every_n_epochs) == 0 and epoch >= self.replacement_start_epoch:
                        x_train = self.replace_pareto(x_train, self.current_samples, run_folder)
                        self.current_samples.clear()

                        if (epoch % 50) == 0:
                            df_train_samples = pd.DataFrame()
                            df_train_samples['LV'] = x_train
                            save_name = 'train_sample_lv_' + self.suffix + '.csv'
                            df_train_samples.to_csv(run_folder + save_name)

            self.epoch += 1

    def replace_pareto(self, x_train, generated_samples, run_folder):
        n_replacements = int(len(x_train) * self.replacement_percent)
        n_kept = len(x_train) - n_replacements

        generated_fitness = self.calculate_fitness(generated_samples)
        self.gen_fitness_avg.append([sum(f)/len(f) for f in zip(*generated_fitness)])
        self.gen_fitness_max.append([max(f) for f in zip(*generated_fitness)])
        self.gen_fitness_min.append([min(f) for f in zip(*generated_fitness)])

        sorted_gen_samples = [val for (_, val) in sorted(zip(generated_fitness, generated_samples), key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)]
        best_gen_samples = self.pareto_front(sorted_gen_samples, n_replacements)

        train_fitness = self.calculate_fitness(x_train)
        self.train_fitness.append([sum(f)/len(f) for f in zip(*train_fitness)])
        sorted_train_samples = [val for (_, val) in sorted(zip(train_fitness, x_train), key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)]
        best_train_samples = sorted_train_samples[:n_kept]

        best_train_samples.extend(best_gen_samples)

        self.pareto_fronts_over_time.append(best_gen_samples)
        self.plot_fitness_progression(run_folder)
        self.plot_pareto_fronts_over_time(run_folder)

        return best_train_samples

    def pareto_front(self, samples, n_replacements):
        pareto_front = []
        for sample in samples:
            dominated = False
            for other in pareto_front:
                if self.dominates(other, sample):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(sample)
            if len(pareto_front) >= n_replacements:
                break
        return pareto_front

    def dominates(self, a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    def plot_fitness_progression(self, run_folder):
        epochs = [(self.replacement_start_epoch + i * self.replace_every_n_epochs) for i in range(len(self.train_fitness))]

        # Extract pIC50, MW, and LogP fitness values for train and generated samples
        train_fitness_pIC50 = [f[2] for f in self.train_fitness]
        train_fitness_mw = [f[0] for f in self.train_fitness]
        train_fitness_logp = [f[1] for f in self.train_fitness]
        gen_fitness_avg_pIC50 = [f[2] for f in self.gen_fitness_avg]
        gen_fitness_avg_mw = [f[0] for f in self.gen_fitness_avg]
        gen_fitness_avg_logp = [f[1] for f in self.gen_fitness_avg]

        # Plot for pIC50
        plt.figure(figsize=(10, 5))
        plt.scatter(epochs, train_fitness_pIC50, label='Train Fitness pIC50', alpha=0.6)
        plt.scatter(epochs, gen_fitness_avg_pIC50, label='Avg Gen Fitness pIC50', alpha=0.6)
        plt.xlabel('Epoch')
        plt.ylabel('pIC50')
        plt.title('Fitness Progression (pIC50)')
        plt.legend()
        save_name = 'pIC50_fitness_progression_' + self.suffix + '.png'
        plt.savefig(os.path.join(run_folder, save_name))
        plt.close()

        # Plot for MW
        plt.figure(figsize=(10, 5))
        plt.scatter(epochs, train_fitness_mw, label='Train Fitness MW', alpha=0.6)
        plt.scatter(epochs, gen_fitness_avg_mw, label='Avg Gen Fitness MW', alpha=0.6)
        plt.xlabel('Epoch')
        plt.ylabel('MW')
        plt.title('Fitness Progression (MW)')
        plt.legend()
        save_name = 'mw_fitness_progression_' + self.suffix + '.png'
        plt.savefig(os.path.join(run_folder, save_name))
        plt.close()

        # Plot for LogP
        plt.figure(figsize=(10, 5))
        plt.scatter(epochs, train_fitness_logp, label='Train Fitness LogP', alpha=0.6)
        plt.scatter(epochs, gen_fitness_avg_logp, label='Avg Gen Fitness LogP', alpha=0.6)
        plt.xlabel('Epoch')
        plt.ylabel('LogP')
        plt.title('Fitness Progression (LogP)')
        plt.legend()
        save_name = 'logp_fitness_progression_' + self.suffix + '.png'
        plt.savefig(os.path.join(run_folder, save_name))
        plt.close()

    def plot_pareto_fronts_over_time(self, run_folder):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        cmap = get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.pareto_fronts_over_time)))

        for i, fronts in enumerate(self.pareto_fronts_over_time):
            if len(fronts) == 0:
                continue
            pIC50_vals, mw_vals, logp_vals = zip(*[self.calculate_fitness([front])[0] for front in fronts])
            ax.scatter(mw_vals, logp_vals, pIC50_vals, color=colors[i], label=f'Epoch {i * self.replace_every_n_epochs}' if i % 10 == 0 else "")

        ax.set_xlabel('MW')
        ax.set_ylabel('LogP')
        ax.set_zlabel('pIC50')
        ax.legend()
        ax.set_title('Pareto Fronts Over Time')
        save_name = 'pareto_fronts_over_time_' + self.suffix + '.png'
        plt.savefig(os.path.join(run_folder, save_name))
        plt.close()

if __name__ == '__main__':
    main_path = '/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/'
    vocab_path = main_path + 'datasets/500k_subset.csv'
    model_path = main_path + 'models/'
    dataset_path = main_path + 'datasets/subsets_100k.csv'
    target_dataset_path = main_path + 'datasets/augmented_dataset.csv'

    df = pd.read_csv(vocab_path)
    selfies = list(df['SELFIES'])
    vocab = Vocabulary(selfies)
    print("Vocab Done!")
    
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

    auto = AE(main_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(model_path + 'AE_model.weights.h5')
    print("AE Done!")

    property = 'pIC50'
    train_df = pd.read_csv(target_dataset_path)
    predictor = Predictor(model_path, property, True, 0.8, vocab, auto, train_df)
    print("Predictor Done!")

    input_dim = 256
    critic_layers_units = [256,256,256]
    critic_lr = 0.0001
    gp_weight = 10
    z_dim  = 64
    generator_layers_units = [128,256,256,256,256]
    generator_batch_norm_momentum = 0.9
    generator_lr = 0.0001
    n_epochs = 1000
    batch_size = 64
    critic_optimizer = 'adam'
    generator_optimizer = 'adam'
    critic_dropout = 0.2
    generator_dropout = 0.2
    n_stag_iters = 50
    print_every_n_epochs = 25
    run_folder = main_path + "GA/"
    suffix = "100k"
    critic_path = model_path + 'critic_GA_' + suffix + '.keras'
    gen_path = model_path + 'generator_GA_' + suffix + '.keras'
    train_df = pd.read_csv(dataset_path)

    gan = ParetoGA(main_path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, generator_optimizer, n_stag_iters, predictor, train_df)
    print("Initialization complete!")
