import getpass
import smtplib
import ssl
from email.message import EmailMessage

sender = input("Gmail address: ").strip()
password = getpass.getpass("Gmail App Password: ")
recipient = input("Test recipient address: ").strip()

message = EmailMessage()
message["Subject"] = "SMTP test"
message["From"] = sender
message["To"] = recipient
message.set_content("Test email: SMTP sending is working.")

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()
        server.login(sender, password)
        refused = server.send_message(message)

    if refused:
        print("SMTP refused the recipient.")
    else:
        print("SMTP accepted the test email. Check the inbox and spam folder.")

except smtplib.SMTPAuthenticationError as error:
    print(f"Authentication failed (SMTP {error.smtp_code}). Check your Gmail App Password.")
except (smtplib.SMTPException, OSError) as error:
    print(f"SMTP check failed: {type(error).__name__}")