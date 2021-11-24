from flask import Flask, render_template, request
import os
import cv2
import tensorflow as tf


categories=['Normal','Tumor']
model = tf.keras.models.load_model('BCDraft01Test.h5')
app = Flask(__name__)

def predic(filepath):
    img_size=224
    img_array=cv2.imread(filepath)
    img_array=cv2.resize(img_array,(img_size,img_size))
    new_array=img_array.reshape(-1,img_size,img_size,3) #[BATCHSIZE,*DIMENSIONS*,COLOR_CHANNELS]
    new_array=new_array/255.0
    prediction=model.predict([new_array])
    if prediction[0][0] >= 0.5:
        percentage = prediction[0][0] * 100
        return categories[1], percentage
    elif prediction[0][0] < 0.5:
         percentage = (1- prediction[0][0]) * 100
         return categories[0], percentage

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    if request.method=='POST':
        imagefile= request.files['imagefile']
        if imagefile:
            image_path = "./images/" + imagefile.filename
            imagefile.save(image_path)
            pred, per=predic(image_path)
            per=format(per, ".2f")
            return render_template('index.html',prediction=pred,percentage=per,image=imagefile.filename)
        else:
            return render_template('index.html',prediction='Upload an image',image=None)
    else:
        return render_template('index.html',prediction='Upload an image',image=None)



    

if __name__ == '__main__':
    app.run(port=3000, debug=True)