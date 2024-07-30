from os import path
import numpy as np
import selfies as sf

class Vocabulary:
     def __init__(self, all_selfies):
         self.max_len = 0
         self.char_to_int = dict()
         self.int_to_char = dict()
         self.unique_chars = set()
         self.vocab_size = 0
         self.create_vocab(all_selfies)
     
     # Returns list of characters/tokens that make up input SELFIES
     def parse_selfies(self, se):
         return list(sf.split_selfies(se))
    
     def create_vocab(self, all_selfies):
        unique_chars = set()
        for se in all_selfies:
            tokens = self.parse_selfies(se)
             
            if len(tokens) > self.max_len:
                self.max_len = len(tokens)

            for t in tokens:
                unique_chars.add(t)
        # Ensures that even longest SELFIES has padding to signal beginning and end of sequence
        self.max_len += 2

         # Padding characters
        unique_chars.add('G')   # start
        unique_chars.add('A')   # padding
        unique_chars = sorted(unique_chars)
        
        vocab = sorted(list(unique_chars))
        self.unique_chars = vocab
        self.vocab_size = len(vocab)

        # Encode
        self.char_to_int = dict()
        for i, char in enumerate(vocab):
            self.char_to_int[char] = i
         
        # Decode
        self.int_to_char = dict()
        for i, char in enumerate(vocab):
            self.int_to_char[i] = char
     
     # Tokenize SELFIES strings
     def tokenize(self, selfies):
        list_tok_selfies = []
        for se in selfies:
            se_tok = ['G']
            se_tok.extend(self.parse_selfies(se))

            # Padding         
            if len(se_tok) < self.max_len:
                 dif = self.max_len - len(se_tok)
                 for _ in range(dif):
                     se_tok.append('A')
                                    
            list_tok_selfies.append(se_tok)

        return list_tok_selfies

     # Encode each char w/ respective index (not one-hot)
     def encode(self, tok_selfies):
         encoded_selfies = []
         for selfie in tok_selfies:
             selfie_idx = []
             for char in selfie:
                 selfie_idx.append(self.char_to_int[char])
             encoded_selfies.append(selfie_idx)
         return encoded_selfies
     
     # Decode numerical index to respective char
     def decode(self, encoded_selfies):
         selfies = []
         for e_selfie in encoded_selfies:
             selfie_chars = []
             for idx in e_selfie:
                 if (self.int_to_char[idx] == 'G'):
                     continue
                 if (self.int_to_char[idx] == 'A'):
                     break 
                 selfie_chars.append(self.int_to_char[idx])
            
             selfie_str = ''.join(selfie_chars)
         
             selfies.append(selfie_str)
         return selfies

     # One hot encode
     def one_hot_encoder(self,selfies_list):
        selfies_one_hot = np.zeros((len(selfies_list),self.max_len, self.vocab_size), dtype = np.int8)
        tokenized = self.tokenize(selfies_list)

        for j, tok in enumerate(tokenized):
           for i, c in enumerate(tok):
               index = self.char_to_int[c]
               selfies_one_hot[j, i, index] = 1

        return selfies_one_hot

     # One hot decode     
     def one_hot_decoder(self, ohe_array):
         all_selfies = []
         for i in range(ohe_array.shape[0]):
             enc_selfie = np.argmax(ohe_array[i,: ,:], axis = 1)
             selfie = ''
             for i in enc_selfie:
                 char = self.int_to_char[i]
                 if char == 'G':
                     continue
                 if char == 'A':
                     break
                 selfie += char
             all_selfies.append(selfie)
         return all_selfies
    
     # Creates output data
     # Shifts everything in input sequence one earlier
     # So, given input X, you are predicting value of next timestep in output Y
     # This is why you add 'G' token in beginning
     def get_target(self, dataX, encode):
          # For one hot encode
          if encode == 'OHE':
              dataY = np.zeros(shape = dataX.shape, dtype = np.int8)
              for i in range(dataX.shape[0]):
                  # [0:last-1] = [1:last]
                  dataY[i,0:-1,:]= dataX[i, 1:, :]
                  # Add extra padding at end
                  dataY[i,-1,self.char_to_int["A"]] = 1
          # For embedding
          elif encode == 'embedding':   
              dataY = [line[1:] for line in dataX]
              for i in range(len(dataY)):
                 dataY[i].append(self.char_to_int['A'])
     
          return dataY