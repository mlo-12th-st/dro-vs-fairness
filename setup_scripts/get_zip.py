'''
Code gets Zip of celebA dataset.
Doesn't currently work for celebA but works for other files on personal drive.
'''


import requests

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
    import sys
    
    # TAKE ID FROM SHAREABLE LINK
    file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM' 
    # DESTINATION FILE ON YOUR DISK
    destination = 'img_align_celeba.zip'
    download_file_from_google_drive(file_id, destination)

# Text file: https://drive.google.com/file/d/1KDVIqhNIukGE3pA3nC8AZ3LFBGB2HnEl/view

# Temp zip: https://drive.google.com/file/d/1e7QbvV5am6N8m6g9fKzY2uPyI7iUcml6/view

# zip of data: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?resourcekey=0-dYn9z10tMJOBAkviAcfdyQ


