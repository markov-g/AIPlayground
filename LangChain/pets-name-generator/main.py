import langchain_helper as lch
import streamlit as st
import os

st.title("Pet Names")
animal_type = st.sidebar.selectbox("What is your pet", ("Cat", "Dog", "Fish", "Hamster"))
pet_color = st.sidebar.text_area(label="What is your pet's color", max_chars=20)

if pet_color:
    lch.load_dotenv()
    llm = lch.create_llms(lch.llm_providers.OLLAMA, 
                      api_key=os.getenv("OPENAI_KEY"),
                      base_url=os.getenv("OLLAMA_BASE_URL"), 
                      model=os.getenv("OLLAMA_MODEL"))
    prompt_template = "I have a {animal_type} pet and I want a cool name for it. It is {animal_color} in color. Suggest me five cool names for my pet."
    variables_values = {'animal_type': animal_type, 'animal_color': pet_color}
    response = lch.get_response(llm, prompt_template, variables_values, output_key="pet_name")
    st.text(response['pet_name'])