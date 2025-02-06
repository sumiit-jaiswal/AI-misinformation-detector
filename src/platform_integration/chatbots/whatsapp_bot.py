from twilio.rest import Client
from fastapi import FastAPI, Request
import requests

app = FastAPI()
client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    user_message = form_data.get("Body")
    sender = form_data.get("From")

    # Call verification API
    response = requests.post("http://localhost:8000/verify", json={"claim": user_message})
    result = response.json()["result"]

    # Send reply
    client.messages.create(
        body=f"Result: {result}",
        from_="whatsapp:+14155238886",
        to=sender
    )
    return {"status": "success"}