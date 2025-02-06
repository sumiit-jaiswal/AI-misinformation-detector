import requests

test_claims = [
    {"text": "The Eiffel Tower is in Paris.", "label": "True"},
    {"text": "The moon is made of cheese.", "label": "False"},
    {"text": "Water boils at 100 degrees Celsius at sea level.", "label": "True"},
    {"text": "Humans have three lungs.", "label": "False"},
    {"text": "The Great Wall of China is visible from space.", "label": "False"},
    {"text": "Albert Einstein developed the theory of relativity.", "label": "True"},
    {"text": "The Amazon Rainforest is located in Africa.", "label": "False"},
    {"text": "A tomato is a fruit, not a vegetable.", "label": "True"},
    {"text": "The Earth orbits the Sun once every 365 days.", "label": "True"},
    {"text": "Sharks are mammals.", "label": "False"}
]

correct = 0

for claim in test_claims:
    response = requests.post("http://localhost:8000/verify", json={"claim": claim["text"]})
    if response.json()["result"] == claim["label"]:
        correct += 1

accuracy = (correct / len(test_claims)) * 100
print(f"Accuracy: {accuracy}%")
