import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 AI-Based Spam Message Detector")
st.write("Classify SMS or Email as **Spam** or **Not Spam (Ham)**")

user_input = st.text_area("Enter your message here:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]

        if prediction == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam (Ham)")
