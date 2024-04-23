from googleapiclient.discovery import build
import pandas as pd
import re
import os
# Function to extract video ID from YouTube URL
def extract_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None

# Function to fetch comments using YouTube Data API
def fetch_youtube_comments(api_key, video_url, max_comments=1000):
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Invalid YouTube video URL")
        return []

    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []

    nextPageToken = None
    total_comments = 0

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()

        for item in response['items']:
            comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            total_comments += 1
            if max_comments and total_comments >= max_comments:
                return comments

        nextPageToken = response.get('nextPageToken')

        if not nextPageToken:
            break

    return comments

def get_data_csv_path(file_name):
    """
    Function to retrieve the full path of a file.
    
    Parameters:
        file_name (str): The name of the file.
        
    Returns:
        str: The full path of the file.
    """
    # Assuming the current working directory is where the script is located
    current_directory = os.getcwd()
    
    # Full path of the file
    file_path = os.path.abspath(file_name)
    
    return file_path