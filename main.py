import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    st.title("Text Generation with chauhan")
    st.write("Enter some text and let chauhan complete it!")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    text = st.text_input("Enter your text:", "Hello my name is")

    if st.button("Generate"):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated text:", generated_text)

if __name__ == "__main__":
    main()
