'''
Source : https://github.com/timesler/facenet-pytorch
'''

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

mtcnn = MTCNN(image_size=(224, 224), margin=(0, 0))

resnet = InceptionResnetV1(pretrained='casia-webface').eval()

