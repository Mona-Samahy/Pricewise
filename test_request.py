import  requests
import base64
import pickle
import json 

file_loc = '11.jpg'

img_bytes = open(file_loc , 'rb' ).read()

en64 = base64.b64encode(   img_bytes  ).decode('utf-8')

# url = 'https://dc0b-41-233-89-253.ngrok-free.app/predict'
url='http://127.0.0.1:8000/predict'

# headers = {"content-type" : "application/json"}


with requests.Session() as s : 
    respone = s.post(url  , json   = {
        'image_data' : en64,  
        'top_n' :  2  
    } ) 

result = respone.json()  
ids =  result['result'] 
print ( ids )