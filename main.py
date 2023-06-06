from tensorflow import keras
import tensorflow as tf
from keras.layers import LSTM, Input,Embedding,Dense,GlobalMaxPooling1D,Flatten
from keras.preprocessing.text import Tokenizer
import json
import nltk
import numpy as np
from tensorflow.keras.models import Model
import string
import nltk
#model=tf.keras.models.load_model("C://Users//Public//model2.h5")
from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn

import string
import nltk
import pandas as pd
#import json
#import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
model=tf.keras.models.load_model("model5.h5",compile=false)
model.compile()
app=FastAPI()

with open('intents (2).json') as intents:
    data=json.load(intents)
class train():
    tags=[]
    inputs=[]
    responses={}
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tags.append(intent['tag'])
    df=pd.DataFrame({'Inputs':inputs,'tags':tags})
    df['Inputs'] = df['Inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation ])
    df['Inputs'] = df['Inputs'].apply(lambda wrd:''.join(wrd))   
    tokenizer=Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(df['Inputs'])
    train=tokenizer.texts_to_sequences(df['Inputs'])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x_train=pad_sequences(train)
    from sklearn.preprocessing import LabelEncoder
    LE=LabelEncoder()

    y_train=LE.fit_transform(df['tags'])  
    input_shape=x_train.shape[1]


    vocabulary=len(tokenizer.word_index)
    print(vocabulary)
    outputlen=LE.classes_.shape[0]
    print(outputlen)
    i=Input(shape=(input_shape))

   


class UserInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True 


 


    
    prediction_input=input("")

    

    
    


    


   


   # prediction_input=np.int.decode(prediction_input,encoding='cp037')
   
    
    #prediction_input = json.dumps(prediction_input)
   

    

   
 


    #prediction_input=pad_sequences([prediction_input],input_shape)
   
    #user_input=pad_sequences([json_data],)
@app.get('/')
async def index():
    return {"hola"}
import random
@app.post('/predict')

async def predict(UserInput:UserInput):

    #pre=json.loads(UserInput.prediction_input)
    #pre =json.dumps(UserInput.prediction_input)
    #new_arr = np.array(UserInput.prediction_input)
    x=train.tokenizer
    y=train.responses
    z=train.input_shape
    #u=train.model  
    k=train.LE
    
    text_p=[]
    prediction_input=[letters.lower() for letters in UserInput.prediction_input if letters not in string.punctuation ]
    prediction_input=''.join(prediction_input)
    text_p.append(prediction_input)
    prediction_input=x.texts_to_sequences(text_p)
   
    prediction_input=np.array(prediction_input).reshape(-1)
    prediction_input=pad_sequences([prediction_input],z)
    prediction=model.predict(prediction_input)
    output=prediction.argmax()
    respones_tag=k.inverse_transform([output])[0]

        
  
    for tag    in   data["intents"]:
            if tag['tag']==tag:
                responses=tag['responses']
    o=(random.choice(y[respones_tag]))
     
    
      
        
            
    

    return {o}
