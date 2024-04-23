import streamlit as st 
import pandas as pd
from method import fetch_youtube_comments, get_data_csv_path
from process import process
import matplotlib.pyplot as plt

# set page configuration
st.set_page_config(
    page_title="Youtube Sentiment Analysis.com"
)

# set title
st.title("Youtube Sentiment Analysis")
st.sidebar.success("Select page above:")
st.subheader("Please enter the link of video")
video_url = st.text_input("Url video")


# Set your YouTube Data API key
api_key = "AIzaSyASoDPeir_7uFoUsX7R6DlvPXa-xtVmgg8"

# Fetch comments
comments = fetch_youtube_comments(api_key, video_url)

# Create a DataFrame with comments
df = pd.DataFrame({'Comments': comments})

# Specify the output file name
output_file = "data.csv"

# Save DataFrame to CSV
df.to_csv(output_file, index=False)
    
# Get the file path
output_file_path = get_data_csv_path(output_file)

# Analyze the data

coll1, coll2, coll3 = st.columns(3)
with coll2:
    Analyzer_button = st.button('Analyze The Video')
st.write("---")

if Analyzer_button:
    try:
        positive, negative, neutral = process(output_file_path)
        column1, column2, column3 = st.columns(3)
        with column2:
            st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image('https://cdn-icons-png.freepik.com/256/5045/5045690.png?semt=ais_hybrid', width=150)
            st.write("Positive Comments:", positive)
        with col2:
            st.image('https://cdn-icons-png.flaticon.com/512/5567/5567134.png', width=150)
            st.write("Negative Comments:", negative)
        with col3:
            st.image('https://cdn-icons-png.flaticon.com/512/1933/1933511.png', width=150)
            st.write("Neutral Comments:", neutral)
        # Plotting the sentiment distribution
        sentiment_data = {'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Count': [positive, negative, neutral]}
        sentiment_df = pd.DataFrame(sentiment_data)

        # Pie chart
        st.write("---")
        st.subheader("Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(sentiment_df['Count'], labels=sentiment_df['Sentiment'], autopct='%1.1f%%', colors=['Lightblue', 'lightcoral', 'lightyellow'], startangle=140)
        ax.axis('equal')  
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        st.write("Please paste the url link to the video!!!")