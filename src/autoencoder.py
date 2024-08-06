import sys
import os
import random
import time

import numpy as np
import pandas as pd
import selfies as sf
from matplotlib import pyplot as plt
from rdkit.Chem import MolFromSmiles
from tensorflow.keras import Model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Concatenate,
    LSTM,
    Bidirectional,
    Dense,
    Input,
    GaussianNoise,
    BatchNormalization,
    Embedding,
)
from tensorflow.keras.optimizers import Adam

# Add the path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Vocabulary import Vocabulary

class Autoencoder:
    """
    Autoencoder class for training and predicting molecular representations.
    """
    def __init__(
        self,
        model_path,
        input_shape,
        latent_dim,
        lstm_units,
        output_dim,
        batch_norm,
        batch_norm_momentum,
        noise_std,
        numb_dec_layer,
        emb_dim,
        vocab_size,
        max_len,
        write_model_arch=False,
    ):
        self.path = model_path
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.noise_std = noise_std
        self.numb_dec_layer = numb_dec_layer
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.write = write_model_arch

        self.build_selfies_to_latent_model()
        self.build_latent_to_states_model()
        self.build_states_to_selfies_model()
        self.build_model()

    def build_selfies_to_latent_model(self):
        """
        Builds the model to convert SELFIES to latent representation.
        """
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        x = Embedding(self.vocab_size, self.lstm_units // 2)(encoder_inputs)

        states_list = []
        states_reversed_list = []
        for i in range(self.numb_dec_layer):
            if self.numb_dec_layer == 1:
                encoder = Bidirectional(
                    LSTM(
                        self.lstm_units // 2,
                        return_state=True,
                        name="encoder" + str(i) + "_LSTM",
                    )
                )
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
            elif i != self.numb_dec_layer - 1:
                encoder = Bidirectional(
                    LSTM(
                        self.lstm_units // 2,
                        return_sequences=True,
                        return_state=True,
                        name="encoder" + str(i) + "_LSTM",
                    )
                )
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
                if self.batch_norm:
                    x = BatchNormalization(
                        momentum=self.batch_norm_momentum, name="BN_" + str(i)
                    )(x)
            else:
                encoder2 = Bidirectional(
                    LSTM(
                        self.lstm_units // 2,
                        return_state=True,
                        name="encoder" + str(i) + "_LSTM",
                    )
                )
                _, state_h2, state_c2, state_h2_reverse, state_c2_reverse = encoder2(x)
                states_list.append(state_h2)
                states_list.append(state_c2)
                states_reversed_list.append(state_h2_reverse)
                states_reversed_list.append(state_c2_reverse)

        complete_states_list = states_list + states_reversed_list
        states = Concatenate(axis=-1, name="concatenate")(complete_states_list)

        if self.batch_norm:
            states = BatchNormalization(
                momentum=self.batch_norm_momentum, name="BN_" + str(i + 1)
            )(states)

        latent_representation = Dense(
            self.latent_dim, activation="relu", name="Dense_relu_latent_rep"
        )(states)

        if self.batch_norm:
            latent_representation = BatchNormalization(
                momentum=self.batch_norm_momentum, name="BN_latent_rep"
            )(latent_representation)

        latent_representation = GaussianNoise(self.noise_std, name="Gaussian_Noise")(
            latent_representation
        )

        self.selfies_to_latent_model = Model(
            encoder_inputs, latent_representation, name="selfies_to_latent_model"
        )

        if self.write:
            sf_to_l = os.path.join(self.path, "selfies_to_latent.txt")
            with open(sf_to_l, "w", encoding="utf-8") as f:
                self.selfies_to_latent_model.summary(
                    print_fn=lambda x: f.write(x + "\n")
                )

    def build_latent_to_states_model(self):
        """
        Builds the model to convert latent representation to initial decoder states.
        """
        latent_input = Input(shape=(self.latent_dim,), name="latent_input")
        decoded_states = []
        for dec_layer in range(self.numb_dec_layer):
            h_decoder = Dense(
                self.lstm_units, activation="relu", name="Dense_h_" + str(dec_layer)
            )(latent_input)
            c_decoder = Dense(
                self.lstm_units, activation="relu", name="Dense_c_" + str(dec_layer)
            )(latent_input)
            if self.batch_norm:
                h_decoder = BatchNormalization(
                    momentum=self.batch_norm_momentum, name="BN_h_" + str(dec_layer)
                )(h_decoder)
                c_decoder = BatchNormalization(
                    momentum=self.batch_norm_momentum, name="BN_c_" + str(dec_layer)
                )(c_decoder)
            decoded_states.append(h_decoder)
            decoded_states.append(c_decoder)

        self.latent_to_states_model = Model(
            latent_input, decoded_states, name="latent_to_states_model"
        )
        if self.write:
            l_to_st = os.path.join(self.path, "latent_to_states.txt")
            with open(l_to_st, "w", encoding="utf-8") as f:
                self.latent_to_states_model.summary(
                    print_fn=lambda x: f.write(x + "\n")
                )

    def build_states_to_selfies_model(self):
        """
        Builds the model to convert hidden and cell states to SELFIES.
        """
        decoder_inputs = Input(shape=self.input_shape, name="decoder_inputs")
        inputs = [decoder_inputs]
        x = decoder_inputs
        for dec_layer in range(self.numb_dec_layer):
            state_h = Input(
                shape=[self.lstm_units], name="Decoded_state_h_" + str(dec_layer)
            )
            state_c = Input(
                shape=[self.lstm_units], name="Decoded_state_c_" + str(dec_layer)
            )
            inputs.append(state_h)
            inputs.append(state_c)
            decoder_lstm = LSTM(
                self.lstm_units,
                return_sequences=True,
                name="Decoder_LSTM_" + str(dec_layer),
            )
            x = decoder_lstm(x, initial_state=[state_h, state_c])
            if self.batch_norm:
                x = BatchNormalization(
                    momentum=self.batch_norm_momentum,
                    name="BN_decoder_" + str(dec_layer),
                )(x)
        outputs = Dense(self.output_dim, activation="softmax", name="Decoder_Dense")(x)

        self.states_to_selfies_model = Model(
            inputs=inputs, outputs=[outputs], name="states_to_selfies_model"
        )
        if self.write:
            st_to_sf = os.path.join(self.path, "states_to_selfies.txt")
            with open(st_to_sf, "w", encoding="utf-8") as f:
                self.states_to_selfies_model.summary(
                    print_fn=lambda x: f.write(x + "\n")
                )

    def build_model(self):
        """
        Combines the encoder and decoder models into a complete Autoencoder model.
        """
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        decoder_inputs = Input(shape=(None, self.input_shape[1]), name="decoder_inputs")
        latent_representation = self.selfies_to_latent_model(encoder_inputs)
        decoded_states = self.latent_to_states_model(latent_representation)
        inputs = [decoder_inputs] + decoded_states
        decoder_outputs = self.states_to_selfies_model(inputs)
        self.model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs,
            name="Autoencoder",
        )

    def load_autoencoder_model(self, path):
        self.model.load_weights(path)
        self.build_sample_model()
        self.build_sm_to_lat()
        print("Weights loaded successfully.")

    def load_encoder_model(self, encoder_path):
        self.model.load_weights(encoder_path)
        print("Encoder weights loaded successfully.")

    def load_decoder_model(self, decoder_path):
        self.model.load_weights(decoder_path)
        print("Decoder weights loaded successfully.")

    def fit_model(self, dataX, dataX2, dataY, epochs, batch_size, optimizer):
        """
        Trains the Autoencoder model.
        """
        self.epochs = epochs
        self.batch_size = batch_size

        if optimizer == "adam":
            self.optimizer = Adam(learning_rate=0.001)
        elif optimizer == "adam_clip":
            self.optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                amsgrad=False,
                clipvalue=3,
            )

        checkpoint_file = os.path.join(self.path, "model--{epoch:02d}.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_file, monitor="val_loss", mode="min", save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        callbacks_list = [checkpoint, reduce_lr, early_stop]

        self.model.compile(optimizer=self.optimizer, loss="categorical_crossentropy")
        results = self.model.fit(
            [dataX, dataX2],
            dataY,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
            callbacks=callbacks_list,
        )

        fig, ax = plt.subplots()
        ax.plot(results.history["loss"], label="Train")
        ax.plot(results.history["val_loss"], label="Val")
        ax.legend()
        ax.set(xlabel="epochs", ylabel="loss")
        figure_path = os.path.join(self.path, "Loss_plot.png")
        fig.savefig(figure_path)
        plt.close()

        self.build_sample_model()
        self.build_sm_to_lat()
        self.model.save_weights(os.path.join(self.path, "AE_model.h5"), save_format='h5')
        self.sample_model.save_weights(os.path.join(self.path, "decoder_model.h5"), save_format='h5')
        self.sm_to_lat_model.save_weights(os.path.join(self.path, "encoder_model.h5"), save_format='h5')

    def build_sample_model(self):
        """
        Converts the trained autoencoder into a latent to SMILES model.
        """
        config = self.states_to_selfies_model.get_config()
        config["input_layers"] = [config["input_layers"][0]]

        idx_list = []
        for idx, layer in enumerate(config["layers"]):
            if "Decoded_state_" in layer["name"]:
                idx_list.append(idx)
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        for layer in config["layers"]:
            idx_list = []
            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoded_state_" in inbound_node[0]:
                        idx_list.append(idx)
            except:
                pass
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        config["layers"][0]["config"]["batch_input_shape"] = (1, 1, self.output_dim)
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True
                layer["config"]["batch_input_shape"] = (1, None, self.output_dim)

        sample_model = Model.from_config(config)
        for layer in sample_model.layers:
            weights = self.states_to_selfies_model.get_layer(layer.name).get_weights()
            sample_model.get_layer(layer.name).set_weights(weights)

        self.sample_model = sample_model
        return config

    def latent_to_selfies(self, latent, vocab):
        """
        Predicts SELFIES from latent representation.
        """
        states = self.latent_to_states_model.predict(np.array([latent]))
        for dec_layer in range(self.numb_dec_layer):
            self.sample_model.get_layer("Decoder_LSTM_" + str(dec_layer)).reset_states(
                states=[states[2 * dec_layer], states[2 * dec_layer + 1]]
            )

        sample_vector = np.zeros(shape=(1, 1, vocab.vocab_size))
        sample_vector[0, 0, vocab.char_to_int["G"]] = 1
        selfies = ""
        for i in range(vocab.max_len - 1):
            pred = self.sample_model.predict(sample_vector)
            idx = np.argmax(pred)
            char = vocab.int_to_char[idx]
            if char == "G":
                continue
            elif char == "A":
                break
            else:
                selfies = selfies + char
                sample_vector = np.zeros((1, 1, vocab.vocab_size))
                sample_vector[0, 0, idx] = 1
        return selfies

    def build_sm_to_lat(self):
        """
        Converts the trained autoencoder into a SELFIES to latent model.
        """
        prediction = self.selfies_to_latent_model.layers[-2].output
        self.sm_to_lat_model = Model(
            inputs=self.selfies_to_latent_model.input, outputs=prediction
        )


def evaluate_reconstruction(real, predicted):
    """
    Determines the percentage of correct molecule reconstruction.
    """
    assert len(real) == len(predicted)
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct += 1
    return correct / len(real) * 100


def evaluate_reconstruction_partial(real, predicted):
    """
    Determines the percentage of characters predicted correctly.
    """
    assert len(real) == len(predicted)
    correct = 0
    total = 0
    for i in range(len(real)):
        index = 0
        while index < len(real[i]) and index < len(predicted[i]):
            if real[i][index] == predicted[i][index]:
                correct += 1
            index += 1
        if len(real[i]) > len(predicted[i]):
            total += len(real[i])
        else:
            total += len(predicted[i])
    return correct / total * 100


def validity(selfies_list):
    """
    Determines the validity of SELFIES predictions.
    """
    total = len(selfies_list)
    valid_selfies = []
    count = 0
    for se in selfies_list:
        sm = sf.decoder(se)
        m = MolFromSmiles(sm)
        if m is not None:
            valid_selfies.append(se)
            count += 1
    perc_valid = count / total * 100
    return valid_selfies, perc_valid


def sample_train_predictions(model, df, vocab, save_path):
    """
    Samples and writes predictions to file.
    """
    selfies = list(df["SELFIES"])
    num_samples = 1
    random.shuffle(selfies)
    sampled_selfies = selfies[:num_samples]

    vocab = Vocabulary(selfies)
    tok_selfies = vocab.tokenize(sampled_selfies)
    enum_selfies = np.array(vocab.encode(tok_selfies))
    ohe_selfies = vocab.one_hot_encoder(sampled_selfies)
    predicted_selfies_probs = model.predict([enum_selfies, ohe_selfies])

    predicted_selfies = []
    for pred_sf in predicted_selfies_probs:
        cur_selfies = ""
        for ohe_sf in pred_sf:
            index = np.argmax(ohe_sf)
            cur_char = vocab.int_to_char[index]
            if cur_char == "G":
                continue
            elif cur_char == "A":
                break
            else:
                cur_selfies += cur_char
        predicted_selfies.append(cur_selfies)

    predictions_path = os.path.join(save_path, "sample_train_predictions.txt")
    with open(predictions_path, "w", encoding="utf-8") as predictions:
        for i in range(len(predicted_selfies)):
            predictions.write("Actu: " + sampled_selfies[i])
            predictions.write("\nPred: " + predicted_selfies[i] + "\n")


if __name__ == "__main__":
    start_time = time.time()

    path = "/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/AE/"
    file = "/gpfs/home/auhhuang/eif4e-inhibitor-discovery/src/datasets/subset_500k.csv"
    
    logistics_path = os.path.join(path, "logistics.txt")
    with open(logistics_path, "w", encoding="utf-8") as run_logistics:
        selfies_file = pd.read_csv(file)
        selfies = list(selfies_file["SELFIES"])
        random.shuffle(selfies)

        vocab = Vocabulary(selfies)
        n_train = int(0.8 * len(selfies))
        selfies_train = selfies[:n_train]
        selfies_test = selfies[n_train:]

        tok_train = vocab.tokenize(selfies_train)
        tok_test = vocab.tokenize(selfies_test)
        encode_train = np.array(vocab.encode(tok_train))
        encode_test = vocab.encode(tok_test)
        x_train = vocab.one_hot_encoder(selfies_train)
        y_train = vocab.get_target(x_train, "OHE")

        run_logistics.write("Vocab Size: " + str(vocab.vocab_size))
        run_logistics.write("\nMax length: " + str(vocab.max_len))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    epochs = 100
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = x_train.shape[1:]
    output_dim = x_train.shape[-1]
    auto = Autoencoder(
        path,
        input_shape,
        latent_dim,
        lstm_units,
        output_dim,
        batch_norm,
        batch_norm_momentum,
        noise_std,
        numb_dec_layer,
        embedding_dim,
        vocab.vocab_size,
        vocab.max_len,
    )
    auto.fit_model(encode_train, x_train, y_train, epochs, batch_size, "adam")

    encode_test = encode_test[:500]
    selfies_test = selfies_test[:500]
    latent_vectors = auto.sm_to_lat_model.predict(encode_test)

    predicted_selfies = []
    for lv in latent_vectors:
        predicted_selfies.append(auto.latent_to_selfies(lv, vocab))

    predictions_path2 = os.path.join(path, "sample_test_predictions2.txt")
    with open(predictions_path2, "w", encoding="utf-8") as example_predictions:
        for i in range(len(selfies_test)):
            example_predictions.write("Actu: " + selfies_test[i])
            example_predictions.write("\nPred: " + predicted_selfies[i] + "\n")

    percent_success = evaluate_reconstruction(selfies_test, predicted_selfies)
    print(percent_success)
    percent_partial_success = evaluate_reconstruction_partial(
        selfies_test, predicted_selfies
    )
    print(percent_partial_success)
    _, percent_valid = validity(predicted_selfies)

    results_path = os.path.join(path, "results.txt")
    with open(results_path, "w", encoding="utf-8") as test_metrics:
        test_metrics.write(
            "Percent Total Successful: " + str(round(percent_success, 4))
        )
        test_metrics.write(
            "\nPercent Partial Successful: " + str(round(percent_partial_success, 4))
        )
        test_metrics.write("\nPercent Valid: " + str(round(percent_valid, 4)))

    with open(logistics_path, "a", encoding="utf-8") as run_logistics:
        run_logistics.write(
            "\nTime (seconds): " + str(round(time.time() - start_time, 3))
        )

    files = os.listdir(path)
    models = [f for f in files if "model--" in f]

    if models:
        epochs = [int(m.split("--")[1].split(".")[0]) for m in models]
        max_epoch = max(epochs)
        for e in epochs:
            if e != max_epoch:
                if int(e / 10) == 0:
                    file_name = "model--0" + str(e) + ".h5"
                else:
                    file_name = "model--" + str(e) + ".h5"
                os.remove(path + file_name)
    else:
        print("No models found in the specified directory.")
