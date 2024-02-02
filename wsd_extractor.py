import os
import zipfile

# Specify the name of the BERT model ZIP file within the 'wsd_model' folder
model_filename = "bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6.zip"

# Specify the path to the 'wsd_model' folder within your project
wsd_model_folder = "wsd_model"

# Generate the full path to the BERT model ZIP file
bert_wsd_pytorch = os.path.join(wsd_model_folder, model_filename)

# Specify the directory where you want to extract the contents
extract_directory = wsd_model_folder

# Generate the path for the extracted folder based on the ZIP file's name
extracted_folder = os.path.join(extract_directory, os.path.splitext(model_filename)[0])


if not os.path.isdir(extracted_folder):
    with zipfile.ZipFile(bert_wsd_pytorch,'r') as zip_ref:
        zip_ref.extractall(extract_directory)
else:
    print(extracted_folder, " is already extracted")




