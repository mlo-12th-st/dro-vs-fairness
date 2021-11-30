# Created by Jared Gridley
# on 11/28/2021
#
# get_data.py
#

import requests


def download_file_from_google_drive(file_id, dest):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    t = get_confirm_token(response)

    if t:
        params = {'id': file_id, 'confirm': t}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "w") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# Sources and destinations:
files = []
#https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
#file_id = '0B7EVK8r0v71pTUZsaXdaSnZBZzg'
#destination = 'image_align_celeba.zip'
#download_file_from_google_drive(file_id, destination)


 # Attempt no. 2

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='0B7EVK8r0v71pTUZsaXdaSnZBZzg', dest_path='/image_align_celeba.zip', unzip=False)


