import os
import uuid

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash
from flask import g
from werkzeug.utils import redirect, secure_filename

from evaluator import get_predictions, _read

app = Flask(__name__)

UPLOAD_FOLDER = 'dicom'
ALLOWED_EXTENSIONS = {'dcm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

@app.route('/')
def home():
    return render_template('home.html')


@app.teardown_request
def teardown_request(exception=None):
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], g.filename))
        # os.remove(os.path.join('static', g.filename.replace('.dcm', '.png')))
    except:
        pass


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predictions', methods=['GET', 'POST'])
def upload_file():
    app.logger.debug(f"request type: {request.method}")

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            app.logger.debug("No file part")
            flash('No file part')
            return redirect('/')
        file = request.files['file']
        app.logger.debug(f"Filename {file.filename}")
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            app.logger.debug("No selected file")
            flash('No selected file')
            return redirect('/')

        if file and allowed_file(file.filename):
            app.logger.debug(secure_filename(file.filename))
            filename = str(uuid.uuid1()).replace('-', '') + '_' + secure_filename(file.filename)
            app.logger.debug(f"path: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.logger.debug("Success file saved")
            preds = get_predictions(filename)[0]
            keys = ['any_prob', 'epidural_prob', 'intraparenchymal_prob', 'intraventricular_prob', 'subarachnoid_prob',
                    'subdural_prob']
            img = _read(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_file = os.path.join('static', filename.replace('.dcm', '.png'))
            plt.imsave(img_file, img)
            g.filename = filename
            app.logger.debug(preds)
            return render_template('predictions.html', any_prob=round(preds[0], 2), epidural_prob=round(preds[1], 2),
                                   intraparenchymal_prob=round(preds[2], 2), intraventricular_prob=round(preds[3], 2),
                                   subarachnoid_prob=round(preds[4], 2), subdural_prob=round(preds[5], 2),
                                   file_name=img_file)
        else:
            app.logger.debug("Error in saving the file")
            return redirect('/')


if __name__ == '__main__':
    app.run()
