# kafka_consumer.py
from kafka import KafkaConsumer
import json
from mongo_handler import insert_claim

consumer = KafkaConsumer(
    "incoming_claims",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

for message in consumer:
    insert_claim(message.value)