import requests
import os
import json


def fetch_games(username, year, month):
    output_dir = "chess_games"
    url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month}"
    # User-Agent for Querying Chess.com
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0 Safari/537.36"
    }
    # Make the API request with headers
    response = requests.get(url, headers=headers)
    # Check the response
    if response.status_code == 200:
       print("Succesfully fetched data")
       games = response.json()
       return games
    else:
        print(f"Failed to fetch games from {username}. Status code: {response.status_code}")