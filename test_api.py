import requests

response = requests.post("http://127.0.0.1:8000/predict", json={"texts": ["I love this so much!", "I hate this terrible thing."]})
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
