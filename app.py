from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid as id
import os
import pandas as pd
from cigna_scikit_models import training_loop

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv']


@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
uuid_id = []
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
      uuid = str(id.uuid1())
      f.save(os.path.join(app.instance_path, 'uploads', uuid))
      return redirect(url_for('column_selector', uuid=uuid)), uuid_id.append(uuid)

@app.route('/selector/<uuid>')#, methods = ['GET', 'POST'])
def column_selector(uuid):
   #if request.method == 'POST':
   orig_df = pd.read_csv(os.path.join(app.instance_path, 'uploads', str(uuid)))
   vars = list(orig_df.columns)
   return render_template('variable_select.html', vars = vars, uuid=uuid)
  
   
@app.route('/chosen_train', methods = ['GET', 'POST'])
def chosen_train():
   uuid = uuid_id.pop()
   if request.method == 'POST':
      #uuid = request.args.get("uuid")
      orig_df = pd.read_csv(os.path.join(app.instance_path, 'uploads', str(uuid)))
      chosen_var = request.form.get('variable', None)
      results, best_params, classifier_best_test = training_loop(orig_df, chosen_var)
      return render_template('simple.html',  tables=[results.to_html(classes='data'), classifier_best_test, best_params.to_html(classes='data')], titles= ['na', 'Training Accuracies', "Classifier with Best Test Accuracy: ", "Best Parameters"])
		
if __name__ == '__main__':
   app.run(debug = True)