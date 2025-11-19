# 📸 Reconhecimento Facial em Vídeos com OpenCV e SVM

![Demonstração do Projeto](demo.gif)

## 📌 Sobre o Projeto

Este é um projeto de Visão Computacional desenvolvido para **detecção e reconhecimento de rostos em vídeos**. O objetivo foi criar um sistema robusto capaz de identificar atores específicos (Bella Ramsey e Pedro Pascal) em cenas de entrevistas, lidando com desafios como variação de ângulo e iluminação.

O pipeline foi construído do zero utilizando um **Detector DNN (Rede Neural Profunda)** do OpenCV para encontrar os rostos e um classificador **SVM (Support Vector Machine)** treinado com embeddings faciais (OpenFace) para o reconhecimento.

O sistema alcançou uma precisão de **90%** no vídeo de teste, processando mais de 2000 frames com performance otimizada.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **OpenCV (DNN):** Utilizando o modelo *ResNet-SSD* para detecção facial robusta.
* **Scikit-learn:** Utilizando o classificador *SVM* para o reconhecimento dos embeddings.
* **OpenFace:** Modelo utilizado para a extração das características faciais (embeddings).
* **NumPy:** Para manipulação de arrays e operações matemáticas.

---

## 🚀 Como Executar

1.  **Clone este repositório:**
    ```bash
    git clone [https://github.com/cicerojr10/reconhecimento-facial-opencv-svm.git](https://github.com/cicerojr10/reconhecimento-facial-opencv-svm.git)
    cd reconhecimento-facial-opencv-svm
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Estrutura de Arquivos Necessária:**
    * Certifique-se de que a pasta `models_pretreinados/` contém os arquivos `.prototxt`, `.caffemodel` e `.t7` (incluídos neste repo).
    * A pasta `dataset_faces/` deve conter as subpastas com as imagens de treino.
    * Adicione um vídeo de teste na pasta `videos_entrada/`.

4.  **Execute o Notebook:**
    * Abra o arquivo `.ipynb` em um ambiente Jupyter ou Google Colab.
    * Execute as células para carregar os modelos, treinar o SVM e processar o vídeo.

---

## 🧠 Desafios de Engenharia

Durante o desenvolvimento, o maior desafio não foi apenas o algoritmo, mas a **Engenharia de Software** envolvida. O projeto exigiu:
* **Gestão de Ambiente:** Resolução de conflitos complexos de dependências e drivers (CUDA/OpenCV).
* **Curadoria de Dados:** A performance do modelo saltou significativamente após a criação de um dataset customizado com alta variabilidade (ângulos e expressões diversos).
* **Otimização:** Implementação de lógica para garantir o processamento eficiente dos frames.

---

## 👤 Autor

* **[Seu Nome]**
* **LinkedIn:** [https://www.linkedin.com/in/seu-perfil/](https://www.linkedin.com/in/cicerojr-techprofessional/)
* **GitHub:** [https://github.com/cicerojr10](https://github.com/cicerojr10)
