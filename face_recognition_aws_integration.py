import boto3
import cv2
import torch
import json
from torchvision import transforms
import os
import imutils
import json
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from shutil import rmtree
import numpy as np

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_recognition_function(key_path):
    # Face extraction
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)
    print("Here")
    # Face recognition
    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    curr_dir = os.getcwd()
    print(curr_dir+'/tmp/data.pt')
    saved_data = torch.load(curr_dir+'/tmp/data.pt')  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        # Save the result name in a file
        with open("/tmp/" + key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return


def handler(event, context):
    try:
        s3 = boto3.client('s3')
        bucket_name = event['bucket_name']
        image_file_name = event['image_file_name']

        # Download the image
        local_image_path = f'/tmp/{image_file_name}'
        s3.download_file(bucket_name, image_file_name, local_image_path)

        results = face_recognition_function(local_image_path)

        if results is not None:
            print(f"Detected face bounding boxes: {results}")
        else:
            print("No faces detected.")

        # Save result to output bucket
        output_file = f'/tmp/{image_file_name.split(".")[0]}.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        s3.upload_file(output_file, '1229855837-output', f'{image_file_name.split(".")[0]}.txt')

        return {'statusCode': 200, 'body': 'Face recognition completed'}
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
