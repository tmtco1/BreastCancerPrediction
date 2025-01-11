import requests

url = "http://127.0.0.1:5000/predict"
image_path = "path_to_your_image.jpg"
files = {"image": open(input(": "), "rb")}

response = requests.post(url, files=files)
print(response.json())
