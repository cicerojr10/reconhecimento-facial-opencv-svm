# 📸 Reconhecimento Facial Híbrido em Vídeos (YOLO + Face-Recognition)

![Demonstração do Projeto](demo.gif)

## 📌 Sobre o Projeto

Este é um projeto de Visão Computacional, desenvolvido como parte do curso de Ciência da Computação, que implementa um pipeline híbrido de alta performance para **detecção e reconhecimento de rostos em vídeos**.

O sistema utiliza o **YOLO (You Only Look Once)** para a detecção robusta de "pessoas" em qualquer ângulo e, em seguida, aplica a biblioteca **`face_recognition` (baseada em dlib)** para identificar rostos específicos dentro das caixas de detecção.

O modelo foi treinado em um dataset customizado (Bella Ramsey e Pedro Pascal) e processa um vídeo de teste, identificando os indivíduos conhecidos e rotulando os desconhecidos.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **OpenCV:** Para manipulação de vídeo e para rodar o detector YOLO.
* **face_recognition (dlib):** Para a extração de embeddings (impressões digitais faciais) e o reconhecimento.
* **NumPy:** Para operações numéricas eficientes.
* **Google Colab:** Como ambiente de desenvolvimento e processamento.

---

## 🚀 Como Executar

1.  **Clone este repositório:**
    ```bash
    git clone [https://github.com/cicerojr10/reconhecimento-facial-opencv-svm.git](https://github.com/cicerojr10/reconhecimento-facial-opencv-svm.git)
    cd reconhecimento-facial-opencv-svm
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (ou .\venv\Scripts\activate no Windows)
    pip install -r requirements.txt
    ```
    *(Nota: a instalação do `face_recognition` e `dlib` pode demorar alguns minutos.)*

3.  **Tenha os arquivos prontos:**
    * O notebook (`02_Reconhecimento_Hibrido_YOLO_FaceRec.ipynb`) já está com o código.
    * Os modelos pré-treinados (YOLO) estão na pasta `models_pretreinados/`.
    * O dataset de treino está em `dataset_faces/`.
    * Adicione um vídeo de teste (ex: `interview_test.mp4`) na pasta `videos_entrada/` (você pode precisar criar esta pasta).

4.  **Execute o Notebook:**
    * Abra o notebook em um ambiente como o Google Colab (com GPU) ou Jupyter Notebook.
    * Execute as células na ordem. O script irá carregar os modelos, treinar (aprender) com as imagens do `dataset_faces` e, em seguida, processar o seu vídeo de entrada, salvando o resultado em `videos_saida/`.

---

## 👤 Autor

* **[Seu Nome Aqui]**
* **LinkedIn:** [https://www.linkedin.com/in/seu-perfil/](https://www.linkedin.com/in/cicerojr-techprofessional/)
* **GitHub:** [https://github.com/cicerojr10](https://github.com/cicerojr10)
