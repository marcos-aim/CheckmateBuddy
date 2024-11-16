import requests


#def fetch_games(username, year, month):
    #url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month}"
url = f"https://api.chess.com/pub/player/magnuscarlsen/games/2014/12"

# User-Agent for Querying Chess.com
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0 Safari/537.36"
}

# Make the API request with headers
response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    print("Successfully fetched data:")
    print(response.json())  # Display the JSON response
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print(response.text)  # Display error details

