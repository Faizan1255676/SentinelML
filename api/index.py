"""
Vercel Serverless Function Handler for SentinelML API.
"""

from src.api.main import app

# Vercel looks for the ASGI 'app' callable
handler = app
