import streamlit as st
import pandas as pd
from method import fetch_youtube_comments, get_data_csv_path
from process import process
import matplotlib.pyplot as plt
from main import api_key


# set title
st.title("Comapare Youtube Sentiment")
st.subheader("Please enter the url of the video")
st.sidebar.success("Select page above:")

# set input area
first_video = st.text_input("First video")
second_video = st.text_input("Second video")
    
# Compare button
cols1,cols2,cols3 = st.columns(3)
with cols2:
    Compare = st.button("Compare the videos")

# fetch comment, write data to csv file    
def main_process(video_url, output_file_name):
    # Fetch comments
    comments = fetch_youtube_comments(api_key, video_url)

    # Create a DataFrame with comments
    df = pd.DataFrame({'Comments': comments})

    # Save DataFrame to CSV
    df.to_csv(output_file_name, index=False)

    # Get the file path
    output_file_path = get_data_csv_path(output_file_name)
    return output_file_path

# Initialize variables outside the if conditions
first_positive = 0
first_negative = 0
first_neutral = 0
second_positive = 0
second_negative = 0
second_neutral = 0

# Video analysis
if Compare:
    first_output_file_path = main_process(first_video, "first_video_comments.csv")
    second_output_file_path = main_process(second_video, "second_video_comments.csv")

    # Process the first video
    first_df = pd.read_csv(first_output_file_path)
    if not first_df.empty:
        first_positive, first_negative, first_neutral = process(first_output_file_path)
    else:
        st.warning("No comments found for the first video.")

    # Process the second video
    second_df = pd.read_csv(second_output_file_path)
    if not second_df.empty:
        second_positive, second_negative, second_neutral = process(second_output_file_path)
    else:
        st.warning("No comments found for the second video.")

    # Show analysis if comments are found
    if not (first_df.empty and second_df.empty):
        column1, column2, column3 = st.columns(3)
        with column2:
            st.subheader("Analysis Results")
        
        st.write("**First Video:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image('https://cdn-icons-png.freepik.com/256/5045/5045690.png?semt=ais_hybrid', width=150)
            st.write("Positive Comments:", first_positive)

        with col2:
            st.image('https://cdn-icons-png.flaticon.com/512/5567/5567134.png', width=150)
            st.write("Negative Comments:", first_negative)

        with col3:
            st.image('https://cdn-icons-png.flaticon.com/512/1933/1933511.png', width=150)
            st.write("Neutral Comments:", first_neutral)
        st.write("**Second Video:**")     
        coll1, coll2, coll3 = st.columns(3)    
        with coll1:
            st.image('https://cdn-icons-png.freepik.com/256/5045/5045690.png?semt=ais_hybrid', width=150)
            st.write("Positive Comments:", second_positive)

        with coll2:
            st.image('https://cdn-icons-png.flaticon.com/512/5567/5567134.png', width=150)
            st.write("Negative Comments:", second_negative)

        with coll3:
            st.image('https://cdn-icons-png.flaticon.com/512/1933/1933511.png', width=150)
            st.write("Neutral Comments:", second_neutral)
        # Define colors for each sentiment
        colors = {'Positive': 'blue', 'Negative': 'red', 'Neutral': 'yellow'}

         # Display first video's sentiment analysis chart
    
        st.subheader("First Video Sentiment Analysis")
        plt.figure(figsize=(8, 6))
        sentiments = ['Positive', 'Negative', 'Neutral']
        counts = [first_positive, first_negative, first_neutral]
        plt.bar(sentiments, counts, color=[colors[sentiment] for sentiment in sentiments])
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Analysis')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Display second video's sentiment analysis chart
   
        st.subheader("Second Video Sentiment Analysis")
        plt.figure(figsize=(8, 6))
        sentiments = ['Positive', 'Negative', 'Neutral']
        counts = [second_positive, second_negative, second_neutral]
        plt.bar(sentiments, counts, color=[colors[sentiment] for sentiment in sentiments])
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Analysis')
        plt.xticks(rotation=45)
        st.pyplot(plt)