import requests
from geopy.geocoders import Nominatim
import streamlit as st
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load chatbot model
model_name = "rebe121314/health_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="your_unique_app_name")
    location = geolocator.geocode(city_name, timeout=10)
    if location:
        return location.latitude, location.longitude
    return None

def get_health_centers(city_name):
    coordinates = get_city_coordinates(city_name)
    if not coordinates:
        return "City not found. Please check your input."
    
    lat, lon = coordinates
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node["amenity"="hospital"](around:10000,{lat},{lon});
    out;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code == 200:
        data = response.json()
        health_centers = []
        
        for element in data.get("elements", []):
            name = element.get("tags", {}).get("name", "Unnamed Health Center")
            center_lat = element.get("lat")
            center_lon = element.get("lon")
            health_centers.append(f"{name} (Lat: {center_lat}, Lon: {center_lon})")
        
        return health_centers if health_centers else "No health centers found in this area."
    else:
        return "Failed to retrieve data from Overpass API."

def healthcare_chatbot(user_input):
    if "symptom" in user_input.lower():
        return "Please consult a doctor for accurate advice."
    elif "appointment" in user_input.lower():
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input.lower():
        return "It's important to take prescribed medicine regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    
    user_input = st.text_input("Ask a healthcare question:")
    if st.button("Get Response"):
        if user_input:
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
        else:
            st.warning("Please enter a message to get a response.")
    
    st.markdown("---")
    st.subheader("Find Health Centers Near You")
    city_name = st.text_input("Enter city name:")
    if st.button("Find Centers"):
        with st.spinner("Fetching health centers..."):
            centers = get_health_centers(city_name)
        st.write("**Health Centers:**", centers)

if __name__ == "__main__":
    main()