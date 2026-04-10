from twilio.rest import Client

def send_whatsapp_alert(account_sid, auth_token, from_number, to_numbers, message_text):
    if not account_sid or not auth_token or not from_number:
        raise ValueError("Twilio credentials or WhatsApp sender number are missing.")

    valid_numbers = [n for n in to_numbers if n]
    if not valid_numbers:
        raise ValueError("No valid emergency WhatsApp numbers configured.")

    client = Client(account_sid, auth_token)

    sent_sids = []
    for number in to_numbers:
        msg = client.messages.create(
            body=message_text,
            from_=from_number,
            to=number
        )
        sent_sids.append(msg.sid)

    return sent_sids