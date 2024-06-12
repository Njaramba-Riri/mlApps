import os 
import numpy as np
import tensorflow as tf
import streamlit as st

# tf.random.set_seed(1)
# np.random.seed(7)

cd = os.getcwd()

pathPy = os.path.join(cd, 'data/txt/mltensor.txt')
pathIs = os.path.join(cd, 'data/txt/island.txt')

modelPy = os.path.join(cd, 'models/pythonML.keras')
modelPy = tf.keras.models.load_model(modelPy)
modelIs = os.path.join(cd, 'models/Island.keras')
modelIs = tf.keras.models.load_model(modelIs)

random = np.random.randint(100, 2500)

def load_data(default_path=True, start_txt='Python Machine Learning', 
         end_txt='zero-padding'):
    if default_path:
        with open(pathPy) as data_path:
            data = data_path.read()
    else:
        with open(pathIs) as data_path:
            data = data_path.read()
     
    start_idx = data.find(start_txt)
    end_idx = data.find(end_txt)
    data = data[start_idx:end_idx] 
       
    char = set(data)
    char_sorted = sorted(char)
    char_array = np.array(char_sorted)
    char2int = {ch:i for i, ch in enumerate(char_sorted)}
    data_encoded = np.array([char2int[ch] for ch in data], dtype=np.int32)
    
    data_encoded = tf.data.Dataset.from_tensor_slices(data_encoded)
    
    return data_encoded, char_array, char2int

def generate_text(default_model=True, prompt_str=None, text_len=random, 
                  max_input_length=50, scale_factor=3):
    if default_model:
        _, char_array, char2int = load_data()
        model = modelPy
        model.reset_states()
    else:
        _, char_array, char2int = load_data(default_path=False,
                                            start_txt='THE MYSTERIOUS ISLAND', 
                                            end_txt='End of the Project Gutenberg')
        model = modelIs
        model.reset_states()
    
    encoded_str = [char2int[ch] for ch in prompt_str]
    encoded_input = tf.reshape(encoded_str, (1, -1))
    
    generated_str = prompt_str
    
    text = st.empty()
    text.text_area("Generated text: ", generated_str)
    
    for _ in range(text_len):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)
        
        scaled_logits = logits * scale_factor
        
        new_char_idx = tf.random.categorical(scaled_logits, num_samples=1)
        new_char_idx = tf.squeeze(new_char_idx)[-1].numpy()
        
        generated_str += str(char_array[new_char_idx])
        
        print(str(char_array[new_char_idx]), end='', flush=True)
        
        new_char_idx_tensor = tf.expand_dims([new_char_idx], 0)
        encoded_input = tf.concat([encoded_input, new_char_idx_tensor], axis=1)
        encoded_input = encoded_input[:, -max_input_length:]
        
        height = len(generated_str) // 2
        
        text.text_area(f"Generated text: { len(generated_str) } of {random + len(prompt_str)}", generated_str, height=height)
    
    return generated_str
