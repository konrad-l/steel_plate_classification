import os
import zipfile

import kaggle

kaggle.api.authenticate()
kaggle.api.competition_download_files('playground-series-s4e3', path='data', force=True, quiet=False)
zip_file_path = 'data/playground-series-s4e3.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('data')
os.remove(zip_file_path)