import cv2  # Importa a biblioteca OpenCV
import numpy as np  # Importa a biblioteca NumPy, que é útil para operações numéricas e manipulação de arrays.

# Cria um classificador de faces usando um arquivo XML de Haarcascade que contém os dados para detectar faces frontais.
classificador = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Cria um classificador para detectar olhos usando um arquivo XML de Haarcascade.
classificadorOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Inicia a captura de vídeo a partir da câmera padrão
camera = cv2.VideoCapture(0)

amostra = 1  # Inicializa um contador de amostras para as imagens capturadas.
numeroAmostras = 25  # Define o número total de amostras que desejamos capturar.
id = input("Digite o seu identificador: ")  # Solicita ao usuário que insira um identificador para as fotos.
largura, altura = 220, 220  # Define as dimensões para redimensionar as imagens capturadas.
print("Capturando as faces...")  # Informa que o processo de captura de faces começou.

# Loop infinito para capturar imagens até que o número de amostras seja alcançado.
while (True):
    conectado, imagem = camera.read()  # Captura um frame da câmera; 'conectado' indica se a captura foi bem-sucedida.
    
    # Converte a imagem capturada de BGR (formato padrão do OpenCV) para escala de cinza.
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Imprime a média dos valores dos pixels da imagem em escala de cinza, que pode indicar a iluminação.
    print(np.average(imagemCinza))
    
    # Detecta faces na imagem em escala de cinza usando o classificador, com ajuste de escala e tamanho mínimo.
    facesDetecatdas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))
    
    # Para cada face detectada, desenha um retângulo ao redor dela na imagem original.
    for (x, y, l, a) in facesDetecatdas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)  # Desenha um retângulo vermelho ao redor da face.

        # Extrai a região da imagem correspondente à face detectada.
        regiao = imagem[y:y + a, x:x + l]
        
        # Converte a região da face para escala de cinza, necessária para detectar olhos.
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        
        # Detecta olhos na região da face usando o classificador de olhos.
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
        
        # Para cada olho detectado, desenha um retângulo ao redor dele.
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)  # Desenha um retângulo verde ao redor do olho.

            # Verifica se a tecla 'q' foi pressionada e se a média dos pixels da imagem é maior que 110.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Se a condição for verdadeira, redimensiona a imagem da face capturada.
                if np.average(imagemCinza) > 110:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    # Salva a imagem redimensionada em um arquivo no formato especificado.
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")  # Informa que a foto foi capturada.
                    amostra += 1  # Incrementa o contador de amostras.

    # Exibe a imagem atual com os retângulos desenhados em uma janela chamada "Face".
    cv2.imshow("Face", imagem)
    
    cv2.waitKey(1)  # Aguarda brevemente por uma tecla antes de continuar o loop.
    
    # Se o número de amostras capturadas for igual ou maior que o número definido, sai do loop.
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")  # Mensagem final indicando que a captura foi concluída.
camera.release()  # Libera a câmera após a captura.
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV.
