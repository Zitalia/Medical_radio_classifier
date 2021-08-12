import os
import shutil

from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import numpy as np

from inference import get_prediction
from commons import format_class_name

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def MultiFiles(files, filename):
    result = []
    for file in files:
        img_bytes = file.read()
        class_name ,class_id = get_prediction(image_bytes=img_bytes)
        result.append(class_name)
        setUploadFolder(class_name)
        file.seek(0)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    x = np.array(result)
    unique = np.unique(x)
    arrayres = []
    for i in range(len(unique)) : 
        arrayres.append({'result': {'class_name' : unique[i],'nbr' : result.count(unique[i])}, 'filename': filename})
    return arrayres

def oneFile(file):
    img_bytes = file.read()
    name_file = file.filename
    class_name ,class_id = get_prediction(image_bytes=img_bytes)
    setUploadFolder(class_name)
    file.seek(0)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    res= [{'result': {'class_name' : class_name,'nbr' : 1}, 'filename': name_file}]
    return res

def setUploadFolder(classname):
    if os.path.isdir(UPLOAD_FOLDER + str(classname)) == False :
        os.mkdir(UPLOAD_FOLDER + str(classname))
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + str(classname)

def Create_zip_file():
    UPLOAD_FOLDER = 'static/uploads/'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    shutil.make_archive('static/uploads/Classified_Xray', 'zip', 'static/uploads/')


def Reset_zip_file():
    if os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            print("redirection")
            return redirect(request.url)
        files = request.files.getlist('files')
        if not files:
            return
        Reset_zip_file()
        if len(files) == 1 :
            file = files[0]
            res = oneFile(file)
        else :
            name_file = f"de {str(len(files))} fichiers"
            res = MultiFiles(files, name_file)
        Create_zip_file()
        return render_template('result.html', data=res)
    return render_template('index.html')


@app.route('/download', methods=['GET', 'POST'])
def download_zip():
    print(request)
    if request.method == 'GET':
        return send_file("static/uploads/Classified_Xray.zip")
    return render_template('index.html')


@app.route('/about')
def hello():
    return render_template('about.html')

if __name__ == '__main__':
    if os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
