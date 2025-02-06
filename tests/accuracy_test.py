# accuracy_test.py
import requests

test_claims = [...]  # Load labeled dataset
correct = 0

for claim in test_claims:
    response = requests.post("http://localhost:8000/verify", json={"claim": claim["text"]})
    if response.json()["result"] == claim["label"]:
        correct += 1

accuracy = (correct / len(test_claims)) * 100
print(f"Accuracy: {accuracy}%")