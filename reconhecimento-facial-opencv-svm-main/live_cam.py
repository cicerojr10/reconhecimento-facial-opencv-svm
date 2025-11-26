import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# --- 1. CONFIGURAÇÃO INICIAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_faces')
MODELS_DIR = os.path.join(BASE_DIR, 'models_pretreinados')

print("[INFO] Carregando modelos de Inteligência Artificial...")

# Caminhos dos arquivos
protoPath = os.path.join(MODELS_DIR, "deploy.prototxt")
# Tenta encontrar o arquivo .txt se o normal não existir
if not os.path.exists(protoPath):
    protoPath = os.path.join(MODELS_DIR, "deploy.prototxt.txt")

modelPath = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
embedderPath = os.path.join(MODELS_DIR, "openface_nn4.small2.v1.t7")

# Verificação de segurança
if not os.path.exists(protoPath) or not os.path.exists(modelPath) or not os.path.exists(embedderPath):
    print("ERRO CRÍTICO: Arquivos de modelo faltando!")
    print(f"Procurando em: {MODELS_DIR}")
    exit()

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(embedderPath)

# --- 2. TREINAMENTO RÁPIDO ---
print("[INFO] Lendo as imagens da pasta 'dataset_faces'...")
knownEmbeddings = []
knownNames = []

if os.path.exists(DATASET_PATH):
    for nome in os.listdir(DATASET_PATH):
        pasta = os.path.join(DATASET_PATH, nome)
        if not os.path.isdir(pasta): continue
        
        print(f"  > Aprendendo: {nome}")
        for arquivo in os.listdir(pasta):
            imagePath = os.path.join(pasta, arquivo)
            image = cv2.imread(imagePath)
            if image is None: continue
            
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    if face.shape[0] < 20 or face.shape[1] < 20: continue
                    
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    knownNames.append(nome)
                    knownEmbeddings.append(vec.flatten())
else:
    print(f"ERRO: Pasta '{DATASET_PATH}' não encontrada.")
    exit()

if len(knownNames) == 0:
    print("ERRO: Nenhuma face encontrada para treinar.")
    exit()

print(f"[INFO] Treinando SVM com {len(knownNames)} rostos...")
le = LabelEncoder()
labels = le.fit_transform(knownNames)
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, labels)
print("[INFO] Sistema PRONTO! Iniciando webcam...")

# --- 3. LOOP DA WEBCAM ---
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (800, 600))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Correção de limites
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20: continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba < 0.65: 
                name = "Desconhecido"
            
            text = f"{name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()