# app.py
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    # Página inicial
    return render_template('index.html')

def gen(camera):
    # Função geradora que envia frames continuamente
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # Rota que o navegador usa para pegar o vídeo
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Roda o servidor acessível na rede local
    app.run(host='0.0.0.0', port=5000, debug=True)