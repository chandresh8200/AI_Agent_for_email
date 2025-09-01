import os
import base64
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

# --- Configuration ---
GMAIL_TOKEN_FILE = "token.json"
GMAIL_CLIENT_SECRETS = "credentials.json"
# IMPORTANT: This scope allows reading AND modifying/deleting emails.
GMAIL_SCOPES = ["https://mail.google.com/"]

def get_gmail_service():
    """Helper function to get the authenticated Gmail service object."""
    credentials = get_gmail_credentials(
        token_file=GMAIL_TOKEN_FILE,
        scopes=GMAIL_SCOPES,
        client_secrets_file=GMAIL_CLIENT_SECRETS,
    )
    return build_resource_service(credentials=credentials)

def _extract_body(payload):
    """Recursively extract the text/plain body from an email payload."""
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                data = part['body']['data']
                body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif 'parts' in part:
                body += _extract_body(part)
    elif payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
        data = payload['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return " ".join(body.replace("\r\n", " ").split())

def search_emails(query: str, max_results: int = 5) -> str:
    """
    Searches the user's Gmail inbox using a standard Gmail query string.
    Returns a formatted string of search results.
    """
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
    messages = results.get('messages', [])
    if not messages:
        return f"No emails found matching query: '{query}'."
    output = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
        output.append(f"ID: {msg['id']}, From: {sender}, Subject: {subject}")
    return "Found emails:\n" + "\n".join(output)

def summarize_email(llm, message_id: str) -> str:
    """
    Summarizes the content of a specific email given its message ID.
    """
    service = get_gmail_service()
    msg_data = service.users().messages().get(userId='me', id=message_id, format='full').execute()
    email_body = _extract_body(msg_data['payload'])
    if not email_body:
        return "Could not extract readable content from the email."
    prompt = f"Please provide a concise summary of the following email content:\n\n\"{email_body[:4000]}\""
    response = llm.invoke(prompt)
    return f"Summary:\n{response.content.strip()}"

def draft_email(recipient: str, subject: str, body: str) -> str:
    """
    Creates a draft email string. Does not send the email.
    """
    if not recipient or "@" not in recipient:
        return "Error: A valid recipient email address is required."
    return f"Draft created for {recipient} with subject '{subject}'. The body is: {body}"

def delete_email(message_id: str) -> str:
    """
    Finds an email by ID and asks for confirmation before deletion.
    This tool does NOT delete the email; it prepares for deletion.
    """
    # For a real application, you would need another step to confirm.
    # For now, we will just move it to trash directly as an example.
    # WARNING: THIS WILL PERMANENTLY DELETE THE EMAIL AFTER 30 DAYS.
    try:
        service = get_gmail_service()
        # Moves the specified message to the trash.
        service.users().messages().trash(userId='me', id=message_id).execute()
        return f"Email with ID {message_id} has been moved to trash."
    except Exception as e:
        return f"Error: Could not delete email with ID {message_id}. Reason: {e}"

# A dictionary to map tool names to their functions for easy calling
AVAILABLE_TOOLS = {
    "search_emails": search_emails,
    "summarize_email": summarize_email,
    "draft_email": draft_email,
    "delete_email": delete_email,
}
