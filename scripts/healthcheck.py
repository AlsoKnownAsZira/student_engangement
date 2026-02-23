"""Tiny health-check helper used by start.ps1"""
import requests

try:
    r = requests.get("http://localhost:8000/health", timeout=2)
    print(r.status_code)
except Exception:
    print(0)
