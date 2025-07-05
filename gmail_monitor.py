from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import time
import subprocess
import sys
import os
import pickle
from google.auth.transport.requests import Request
import re
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
def search_email(service, job_id):
    query = f'subject:"SLURM Array Summary Job_id={job_id}"'
    result = service.users().messages().list(userId='me', q=query, maxResults=1).execute()
    messages = result.get('messages', [])
    return messages
def get_subject(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['Subject']).execute()
    headers = msg['payload']['headers']
    for header in headers:
        if header['name'] == 'Subject':
            return header['value']
    return ""

def search_email_all(service, subject_prefix):
    """Return every message with subject_prefix, across all pages."""
    all_msgs = []
    page_token = None
    while True:
        resp = service.users().messages().list(
            userId='me',
            q=f'subject:"{subject_prefix}"',
            pageToken=page_token,
            maxResults=100
        ).execute()
        all_msgs.extend(resp.get('messages', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return all_msgs


def monitor(job_id_array):
    print("looking for: ",job_id_array)
    service = authenticate()
    seen_ids = set()
    print("Watching inbox for Slurm emails marked COMPLETED...")
    # while job_id_array:
    #     for jid in job_id_array:
    #         msgs = search_email(service, jid)
    #         if msgs:
    #             msg_id = msgs[0]['id']
    #             if msg_id not in seen:
    #                 seen.add(msg_id)
    #                 print(f"found job {jid}")
    #                 job_id_array.remove(jid)
    #     if job_id_array:
    #         print("waiting for ", job_id_array)
    #         time.sleep(5)




    while job_id_array: 
        messages = search_email_all(service, 'SLURM Array Summary Job_id=')
        for msg in messages:
            msg_id = msg['id']
            if msg_id not in seen_ids:
                subject = get_subject(service, msg_id)
                if "Ended"in subject and any(job_id in subject for job_id in job_id_array):
                    # seen_ids.add(msg_id)
                    # print(f"New COMPLETED job email: {subject}")
                    match = re.search(r"\((\d+)\)", subject)
                    job_id = match.group(1)
                    print(f"Found job "+ job_id)
                    # subprocess.run(["./automated_pipeline.sh", "download", job_id])
                    job_id_array.remove(job_id)
                    if not job_id_array:
                        return
        print("loop monitor")
        print("job id array: ", job_id_array)
        time.sleep(5)
    return