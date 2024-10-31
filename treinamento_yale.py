import cv2
import os
import numpy as np
from PIL import Image

#eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
#fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=4, grid_x=8, grid_y=8, threshold=50)

def getImagemComId():
    caminhos = [os.path.join('yalefaces/treinamento', f) for f in os.listdir('yalefaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')
       imagemNP = np.array(imagemFace, 'uint8')
       id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
       ids.append(id)
       faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...")
#eigenface.train(faces, ids)
#eigenface.write('classificadorEigenYale.yml')

#fisherface.train(faces, ids)
#fisherface.write('classificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHYale.yml')

print("Treinamento realizado")