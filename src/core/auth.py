from __future__ import annotations
import os
from dotenv import load_dotenv
from .broker import Broker


def broker_from_env() -> Broker:
    load_dotenv()
    api_key = os.getenv("UPSTOX_API_KEY")
    api_secret = os.getenv("UPSTOX_API_SECRET")
    access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not (api_key and api_secret and access_token):
        raise ValueError("Missing UPSTOX_API_KEY/UPSTOX_API_SECRET/UPSTOX_ACCESS_TOKEN")
    return Broker(api_key, api_secret, access_token)
