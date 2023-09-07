'''
Source : https://github.com/timesler/facenet-pytorch
'''

from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=(160, 160), margin=(0, 0))

resnet = InceptionResnetV1(pretrained='imagenet').eval()
