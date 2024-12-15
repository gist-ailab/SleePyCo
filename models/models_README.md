Quick explanation of file tree: \
For each model we have a subfolder: \
- root \
&nbsp;&nbsp;    - cnn \
&nbsp;&nbsp;    - transformer \
&nbsp;&nbsp;    - ... \
Each subfolder has a model file and a backbone file.\
The model file is the actual DL model. \
The backbone file shows how the model was originally used and can be changed arbitrarily (especially forward())
