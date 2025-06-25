from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import time
import subprocess
import sys
import os 
import pickle 
from google.auth.transport.requests import Request


# Gmail API scope to read emails
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate():
    creds = None
    # check if token is there already
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token:
            creds = pickle.load(token)

    # if no token first check if we can refresh an expired one
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        
        # if we cant do the entire login sequence again
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentialslance.json', SCOPES)
            creds = flow.run_local_server(port=0)
    
    # save the token for future use
    with open('token.pkl', 'wb') as token:
        pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def search_email(service, subject_prefix):
    query = f'subject:"{subject_prefix}"'
    result = service.users().messages().list(userId='me', q=query).execute()
    messages = result.get('messages', [])
    return messages

def get_subject(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['Subject']).execute()
    headers = msg['payload']['headers']
    for header in headers:
        if header['name'] == 'Subject':
            return header['value']
    return ""

def monitor(job_id):
    service = authenticate()
    seen_ids = set()

    print("Watching inbox for Slurm emails marked COMPLETED...")

    while True:
        job_id_header = "Slurm Array Summary Job_id="+job_id
        messages = search_email(service, 'SLURM Array Summary Job_id=')

        for msg in messages:
            msg_id = msg['id']
            if msg_id not in seen_ids:
                subject = get_subject(service, msg_id)
                if "Ended"in subject and job_id in subject:
                    seen_ids.add(msg_id)
                    print(f"New COMPLETED job email: {subject}")
                    
                    subprocess.run(["./automated_pipeline.sh", "download", job_id])
                    return 
        
        print("loop monitor")
        time.sleep(30)

if __name__ == '__main__':
    main()

