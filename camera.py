import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

class VideoCamera(object):
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = os.path.join(self.BASE_DIR, 'dataset_faces')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'models_pretreinados')

        print("[S.H.I.E.L.D.] Inicializando Protocolos de Segurança...")
        
        # Carregar Modelos
        protoPath = os.path.join(self.MODELS_DIR, "deploy.prototxt")
        if not os.path.exists(protoPath): protoPath += ".txt"
        modelPath = os.path.join(self.MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        embedderPath = os.path.join(self.MODELS_DIR, "openface_nn4.small2.v1.t7")

        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.embedder = cv2.dnn.readNetFromTorch(embedderPath)

        # Treinamento Automático
        print("[S.H.I.E.L.D.] Carregando banco de dados de Agentes...")
        knownEmbeddings = []
        knownNames = []

        if os.path.exists(self.DATASET_PATH):
            for nome in os.listdir(self.DATASET_PATH):
                pasta = os.path.join(self.DATASET_PATH, nome)
                if not os.path.isdir(pasta): continue
                
                for arquivo in os.listdir(pasta):
                    imagePath = os.path.join(pasta, arquivo)
                    image = cv2.imread(imagePath)
                    if image is None: continue
                    
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.detector.setInput(blob)
                    detections = self.detector.forward()

                    if len(detections) > 0:
                        i = np.argmax(detections[0, 0, :, 2])
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:
                            face = image # Simplificado para treino rápido
                            # (Lógica de recorte omitida para brevidade do treino, mas idealmente recorta-se a face)
                            # No treino real, garanta que as fotos sejam só do rosto ou use a lógica completa
                            # Aqui assumimos que suas fotos de treino já são boas.
                            
                            # Re-adicionando recorte para garantir qualidade se a foto for grande
                            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                            (startX, startY, endX, endY) = box.astype("int")
                            face_crop = image[startY:endY, startX:endX]
                            
                            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20: continue

                            faceBlob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                            self.embedder.setInput(faceBlob)
                            vec = self.embedder.forward()
                            knownNames.append(nome)
                            knownEmbeddings.append(vec.flatten())

        if len(knownNames) > 0:
            self.le = LabelEncoder()
            labels = self.le.fit_transform(knownNames)
            self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
            self.recognizer.fit(knownEmbeddings, labels)
            self.is_trained = True
            print(f"[S.H.I.E.L.D.] Sistema Ativo. {len(knownNames)} Agentes monitorados.")
        else:
            self.is_trained = False

        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None

        frame = cv2.resize(frame, (800, 600))
        (h, w) = frame.shape[:2]

        # Filtro Azulado (Tech look)
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE) # Opcional: deixa cinza/azulado

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Desenhar linhas de HUD fixas (Mira central)
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
        cv2.putText(frame, "SCANNING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                face = frame[startY:endY, startX:endX]
                
                # Default: Intruso
                name_display = "UNKNOWN SUBJECT"
                id_display = "ID: 000-00-0000"
                status_display = "ACCESS DENIED"
                color = (0, 0, 255) # Vermelho
                
                if self.is_trained and face.shape[0] > 20 and face.shape[1] > 20:
                    try:
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        self.embedder.setInput(faceBlob)
                        vec = self.embedder.forward()

                        preds = self.recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba_val = preds[j]
                        
                        # Lógica SHIELD: Só libera se for muito parecido com o Agente
                        if proba_val > 0.75:
                            raw_name = self.le.classes_[j].upper()
                            if "PEDRO" in raw_name:
                                name_display = "AGENT PASCAL"
                                id_display = "ID: SHD-8942"
                                status_display = "LEVEL 7 CLEARANCE"
                                color = (255, 255, 0) # Ciano/Azul claro (BGR) -> (255, 255, 0) é Ciano no OpenCV? Não, (255,255,0) é Ciano.
                                color = (255, 255, 0) # Ciano
                        else:
                            # Confiança baixa ou desconhecido
                            pass # Mantém vermelho
                    except:
                        pass

                # --- DESENHO HUD FUTURISTA ---
                
                # Caixa com cantos
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
                # Cantos grossos
                l = 30
                t = 4
                cv2.line(frame, (startX, startY), (startX + l, startY), color, t)
                cv2.line(frame, (startX, startY), (startX, startY + l), color, t)
                cv2.line(frame, (endX, endY), (endX - l, endY), color, t)
                cv2.line(frame, (endX, endY), (endX, endY - l), color, t)
                
                cv2.line(frame, (startX, endY), (startX + l, endY), color, t)
                cv2.line(frame, (startX, endY), (startX, endY - l), color, t)
                cv2.line(frame, (endX, startY), (endX - l, startY), color, t)
                cv2.line(frame, (endX, startY), (endX, startY + l), color, t)

                # Texto lateral ou superior
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Fundo preto para texto
                cv2.rectangle(frame, (endX + 5, startY), (endX + 250, startY + 70), (0,0,0), -1)
                cv2.rectangle(frame, (endX + 5, startY), (endX + 250, startY + 70), color, 1)
                
                cv2.putText(frame, name_display, (endX + 10, startY + 20), font, 0.5, color, 1)
                cv2.putText(frame, id_display, (endX + 10, startY + 40), font, 0.4, (200,200,200), 1)
                cv2.putText(frame, status_display, (endX + 10, startY + 60), font, 0.4, color, 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()