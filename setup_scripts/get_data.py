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

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


file_id = '0B6lEo20DNMISIJDSJDVBMENXbkE'
destination = './data/'
download_file_from_google_drive(file_id, destination)
