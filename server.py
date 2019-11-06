import os
from flask import Flask,render_template,request,url_for,redirect,send_file
from skimage.transform import resize
from imageio import imread
from keras.models import model_from_json
import numpy as np

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template('HOMEPAGE.html')
    if request.method=='POST':
        image_file=request.files['image']
        filepath = os.path.join('IMAGEFORPRO', image_file.filename)
        image_file.save(filepath)
        return redirect(url_for('RESULT',filename1=image_file.filename))

@app.route('/images/<filename>',methods=['GET'])
def images(filename):
    return send_file(os.path.join('IMAGEFORPRO', filename))

@app.route('/RESULT/<filename1>')
def RESULT(filename1):
    json_file=open(r'C:\Users\Lenovo\projectsem\Model .json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)
    model.load_weights(r"C:\Users\Lenovo\projectsem\model.h5")
    image_path=os.path.join('IMAGEFORPRO',filename1)
    image_url=url_for('images',filename=filename1)
    image_initial=imread(image_path)
    image_final=resize(image_initial,(32,32,3))
    prob=model.predict(np.array([image_final,]))
    options=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    index = np.argsort(prob[0,:])
    OBJ=options[index[9]]
    return render_template('RESULT.html',OBJ=OBJ,image_url=image_url)
    

if __name__=='__main__':
    app.run('127.0.0.1')	