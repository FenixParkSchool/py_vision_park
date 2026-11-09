# Roteiro para o computador do Senac (treino e avaliação)

Objetivo: treinar detector e OCR com o RodoSol-ALPR, medir e trazer de volta **só** os modelos e as métricas.

## 1. Preparar
```
git clone <repo> py_vision_park && cd py_vision_park
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt     # inclui TensorFlow
.venv\Scripts\python -m pytest tests/unit -q          # deve passar antes de treinar
```
Anote `python --version`, `tensorflow` e `keras` instalados (a ficha do modelo registra as duas últimas).
O dataset deve estar em `data/rodosol/` (pastas com `.jpg` + `.txt` de mesmo nome). Não vai para o git.

## 2. Treinar (CPU, minutos)
```
.venv\Scripts\python -m training.train_detector --per-folder 600 --epochs 12
.venv\Scripts\python -m training.train_ocr --per-folder 1500 --epochs 15
```
Saída em `models/`: `detector.keras`, `ocr.keras` e uma ficha `.json` ao lado de cada um
(data, sha256, versões de TensorFlow/Keras, parâmetros, métricas de validação, tamanho dos conjuntos).

## 3. Avaliar no conjunto de teste
```
.venv\Scripts\python -m training.evaluate            # ou --limit 500 para um teste rápido
```
Grava `models/avaliacao.json` com, por tipo de veículo/layout (carro e moto, Brazilian e Mercosul):
taxa de detecção (IoU >= 0.5), acerto do OCR isolado (placa inteira e por caractere),
acerto ponta a ponta, % de leituras com formato válido e tempo médio por etapa (ms).

## 4. Trazer de volta
Copie para esta máquina, em `models/` (fora do git): `detector.keras`, `ocr.keras`, as duas fichas `.json` e `avaliacao.json`.
Confira o hash: `sha256` da ficha deve bater com o arquivo `.keras`.

## 5. Usar
No `.env`: `RECOGNITION_MODE=tf` (e `DETECTOR_MODEL_PATH`/`OCR_MODEL_PATH` se mudar os caminhos).
O serviço registra no log o sha256 e a data de cada modelo. Se faltar arquivo, `/health/ready` responde 503 com o motivo.

## 6. Decidir com os números (não com palpite)
- `MIN_CONFIDENCE`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE`: ajustar olhando `avaliacao.json`.
- Se `e2e_formato_valido` for baixo em algum grupo (principalmente moto, de duas linhas), registrar no contexto antes de qualquer correção de caracteres.
- Resolução da webcam: o detector foi treinado com frames 1280x720 reduzidos a 384 de largura; testar com a webcam real antes de confiar nos números.
