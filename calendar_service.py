import os
import json
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import traceback
import streamlit as st

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = 'token.json'
CREDENTIALS_PATH = 'credentials.json'

def get_credentials():
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(TOKEN_PATH, "w") as token:
                    token.write(creds.to_json())

            return creds

        except Exception:
            if os.path.exists(TOKEN_PATH):
                os.remove(TOKEN_PATH)
            return None

    return None

def is_connected():
    creds = get_credentials()
    return creds is not None and creds.valid

def disconnect():
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)
        return True
    return False

def save_client_credentials(client_id, client_secret):
    client_config = {
        "web": {
            "client_id": client_id,
            "project_id": "nitr-assistant",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost:8501/"]
        }
    }
    with open(CREDENTIALS_PATH, 'w') as f:
        json.dump(client_config, f, indent=4)

def has_client_credentials():
    return os.path.exists(CREDENTIALS_PATH)

def connect_google_calendar():
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_PATH,
        SCOPES
    )

    creds = flow.run_local_server(
        port=8502,
        open_browser=True
    )

    with open(TOKEN_PATH, "w") as token:
        token.write(creds.to_json())

    return True

def get_calendar_service():
    creds = get_credentials()
    if not creds:
        raise ValueError("Google Calendar is not connected. Please authorize it first.")
    return build('calendar', 'v3', credentials=creds)

def list_events(calendar_id="primary", max_results=10, time_min=None, time_max=None):
    try:
        service = get_calendar_service()
        if not time_min:
            time_min = datetime.utcnow().isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return events_result.get('items', [])
    except Exception as e:
        return f"Error listing events: {str(e)}"

def create_event(summary, start_time, end_time, description=None, location=None, calendar_id="primary"):
    try:
        service = get_calendar_service()
        event_body = {
            'summary': summary,
            'start': {'dateTime': start_time},
            'end': {'dateTime': end_time}
        }
        if description:
            event_body['description'] = description
        if location:
            event_body['location'] = location
            
        event = service.events().insert(calendarId=calendar_id, body=event_body).execute()
        return f"Event '{summary}' created successfully: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error creating event: {str(e)}"

def delete_event(event_id, calendar_id="primary"):
    try:
        service = get_calendar_service()
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return "Event deleted successfully."
    except Exception as e:
        return f"Error deleting event: {str(e)}"

def update_event(event_id, summary=None, start_time=None, end_time=None, description=None, location=None, calendar_id="primary"):
    try:
        service = get_calendar_service()
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        
        if summary:
            event['summary'] = summary
        if start_time:
            event['start'] = {'dateTime': start_time}
        if end_time:
            event['end'] = {'dateTime': end_time}
        if description is not None:
            event['description'] = description
        if location is not None:
            event['location'] = location
            
        updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()
        return f"Event updated successfully: {updated_event.get('htmlLink')}"
    except Exception as e:
        return f"Error updating event: {str(e)}"