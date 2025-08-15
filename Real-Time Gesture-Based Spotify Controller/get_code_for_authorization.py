import urllib.parse
import os
from dotenv import load_dotenv
import json
import os
import base64
from requests import post


load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID")
if client_id is None:
    raise ValueError("SPOTIFY_CLIENT_ID not found in .env file!")

print("Client ID:", client_id)  
scopes = "user-modify-playback-state user-read-playback-state"

auth_url = "https://accounts.spotify.com/authorize"
redirect_uri = "http://127.0.0.1:8888/callback" 
params = {
    "client_id": client_id,
    "response_type": "code",
    "redirect_uri": redirect_uri,
    "scope": scopes
}

url = auth_url + "?" + urllib.parse.urlencode(params)
print("Go to this URL to authorize:", url)

code = "AQBXC6WRbeciQDEbdLUekVKLtqa61MIPIAYAT1S9dN2Fh56zTPscwLGNEGArmcUoyWSpEfCgYOY-lzFHxJ6M_c_1f7JL1o1P3PaC615MLIK29cYRZye81Zk9IZY6dsCAHMsJIiahPRS7TkCSIp-Z2-5ObeJ0YJtlvkNnpDUMZ_C4gsDZRG0nccM_V9KXjPBhmNCZIcm36D2juIj1shVgG66LxqLbKEUgUDw6_s7TmYgjLd7VqjEr5g"

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = "http://127.0.0.1:8888/callback"


def get_token():
    auth_string = client_id + ':' + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth_base64}"}
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }

    result = post(url, headers=headers, data=data).json()
    print(result)
    access_token = result["access_token"]
    refresh_token = result["refresh_token"]

    print("Access Token:", access_token)
    print("Refresh Token:", refresh_token)

    if access_token:
        return access_token
    else:
        raise Exception("Failed to retrieve access token from Spotify API")

token = get_token()
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}