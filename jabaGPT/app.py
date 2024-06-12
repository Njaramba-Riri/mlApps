import streamlit as st
from generate import generate_text

st.set_page_config(page_title="LangGPT")

st.title("Limpueza: A custom RNN that generates text.")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/globe:")
st.text("Prompt the model with a custom text.")

options = st.radio('Select model', ('Model 1', 'Model 2'))
prompt = st.text_input("Enter your prompt here: ")

if options == 'Model 1':
    
    if st.button("Generate Text"):
        if prompt:
            generate_text(prompt_str=prompt)
        else:
            st.write("Please enter a prompt string.")

elif options == 'Model 2':    
    if st.button("Generate Text"):
        if prompt:
            generate_text(default_model=False, prompt_str=prompt)
        else:
            st.write("Please enter a prompt string.")
