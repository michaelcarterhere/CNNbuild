import base64

with open("digit.png", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    print(base64_string)