import cv2  # Importa a biblioteca OpenCV
import os  # Importa a biblioteca os, que permite interagir com o sistema operacional
import numpy as np  # Importa a biblioteca NumPy

# Cria um reconhecedor de faces usando o método EigenFaces com 50 componentes principais e um limite de 2 para a distância de reconhecimento.
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=2)

# Cria um reconhecedor de faces usando o método FisherFaces.
fisherface = cv2.face.FisherFaceRecognizer_create()

# Cria um reconhecedor de faces usando o método LBPH (Local Binary Patterns Histograms).
lbph = cv2.face.LBPHFaceRecognizer_create()

# Função para obter imagens e seus respectivos IDs a partir da pasta 'fotos'.
def getImagemComId():
    # Cria uma lista de caminhos para as imagens na pasta 'fotos'.
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    
    faces = []  # Inicializa uma lista para armazenar as imagens das faces.
    ids = []    # Inicializa uma lista para armazenar os IDs correspondentes às faces.
    
    # Itera sobre cada caminho de imagem na lista de caminhos.
    for caminhoImagem in caminhos:
        # Lê a imagem e converte para escala de cinza.
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        
        # Extrai o ID da imagem a partir do nome do arquivo. O ID está na segunda parte do nome (ex: pessoa.1.jpg).
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        
        # Adiciona o ID e a imagem da face às listas correspondentes.
        ids.append(id)
        faces.append(imagemFace)
        
        # As linhas abaixo estão comentadas, mas poderiam ser usadas para mostrar as faces durante a leitura.
        # cv2.imshow("Faces", imagemFace)  # Exibe a imagem da face em uma janela.
        # cv2.waitKey(10)  # Aguarda 10 milissegundos entre as imagens.
    
    return np.array(ids), faces  # Retorna os IDs como um array NumPy e a lista de faces.

# Chama a função getImagemComId e armazena os resultados nas variáveis ids e faces.
ids, faces = getImagemComId()

# As linhas abaixo estão comentadas, mas poderiam ser usadas para visualizar os IDs e faces.
# print(ids)  # Imprime os IDs das faces.
# print(faces)  # Imprime as faces capturadas.

print("treinando...")  # Informa que o treinamento dos reconhecedores está prestes a começar.

# Treina o reconhecedor EigenFace com as faces e seus IDs.
eigenface.train(faces, ids)

# Salva o classificador treinado EigenFace em um arquivo YAML.
eigenface.write('classificadorEigen.yml')

# Treina o reconhecedor FisherFace com as faces e seus IDs.
fisherface.train(faces, ids)

# Salva o classificador treinado FisherFace em um arquivo YAML.
fisherface.write('classificadorFisher.yml')

# Treina o reconhecedor LBPH com as faces e seus IDs.
lbph.train(faces, ids)

# Salva o classificador treinado LBPH em um arquivo YAML.
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")  # Informa que o treinamento foi concluído com sucesso.
