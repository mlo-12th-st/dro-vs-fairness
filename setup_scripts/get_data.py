'''
Code gets Zip of celebA dataset, then unzips the data

To get celebA dataset run: python3 get_data.py 1nNGRZOl9X4ryeCkEKMBbxNKeCHOp4ShU ../data/celeba/img_align_celeba.zip
'''

import sys
import os
import requests
from zipfile import ZipFile



def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    
    if(len(sys.argv) <= 3):
        print('Invalid Arguments!')
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]  
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2] 

        spot = sys.argv[2].split('/')[:-1]
        s = '/'
        spot = s.join(spot)

        if not os.path.exists(spot):
            os.makedirs(spot)
        
        download_file_from_google_drive(file_id, destination)

        if(len(sys.argv) == 3):
            with ZipFile(sys.argv[2], 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(spot)
            os.remove(sys.argv[2])
        

            
