# py_vision_park — microsserviço ALPR

Lê frames de uma câmera (arquivo, RTSP ou USB), detecta a placa, faz OCR e envia o evento
ao backend (`POST {BACKEND_URL}/eventos-alpr`). Hoje o detector e o OCR são **fakes**
por padrão (`RECOGNITION_MODE=fake`). Com `RECOGNITION_MODE=tf` usa os modelos Keras de `models/` (treino e avaliação: [docs/senac.md](docs/senac.md)).

## Rodar
```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt   # TensorFlow só é necessário para treino/modelos reais
copy .env.example .env                              # preencha BACKEND_API_KEY (mínimo 32 caracteres, igual ao ALPR_API_KEY do Back_park)
.venv\Scripts\python iniciar.py
```
Testes: `.venv\Scripts\python -m pytest tests/unit -q`

## Configuração (`.env`)
| Variável | Padrão | Significado |
|---|---|---|
| `CAMERA_SOURCE` | `data/teste.mp4` | arquivo, URL RTSP ou índice da webcam |
| `CAMERA_ID` | `portaria-1` | identificador enviado no evento |
| `BACKEND_URL` | `http://localhost:3000` | base do Back_park (sem `/api`) |
| `BACKEND_API_KEY` | vazio | enviada no header `X-API-Key`; obrigatória |
| `MIN_CONFIDENCE` | `0.6` | confiança mínima da detecção (provisório) |
| `DEDUP_SECONDS` | `30` | janela para não repetir a mesma placa (provisório) |
| `SAMPLE_INTERVAL_SECONDS` | `1.0` | intervalo entre frames processados (provisório) |
| `BACKEND_TIMEOUT` | `5` | timeout por requisição, em segundos |
| `BACKEND_MAX_RETRIES` | `3` | tentativas extras após falha (backoff 0.5s, 1s, 2s...) |
| `RECOGNITION_MODE` | `fake` | `fake` ou `tf` (modelos reais) |
| `DETECTOR_MODEL_PATH` / `OCR_MODEL_PATH` | `models/detector.keras` / `models/ocr.keras` | usados só em `tf`; modelo ausente = `/health/ready` 503 |
| `DETECTOR_THRESHOLD` | `0.5` | limiar do mapa de probabilidade do detector |
| `OCR_MIN_CONFIDENCE` | vazio | se vazio, usa `MIN_CONFIDENCE` também no OCR |

## Health
- `GET /health/live`: o processo está de pé.
- `GET /health/ready`: 200 com contadores (`frames`, `events_sent`, `duplicates`, `send_failures`, `invalid_plates`); 503 se a câmera não abriu ou o worker parou.

## Contrato com o backend
Body JSON camelCase e estrito (campo extra = 400): `eventId`, `cameraId`, `plateText`, `detectionConfidence`, `ocrConfidence`, `capturedAt` (ISO 8601 com offset).
Idempotência por `eventId`: 201 novo, 200 `duplicado:true`, 409 corrida (reenvia).
Reenvia em rede/timeout/409/429/5xx (429 respeita `Retry-After`, teto de 30s); não reenvia em 400/401/503.

## Políticas
- **Perda de eventos:** se o backend rejeitar ou ficar indisponível após as tentativas, o evento é logado e **descartado**; não há fila nem banco. O worker continua.
- **Placa inválida:** só formatos Mercosul (`ABC1D23`) e antigo (`ABC1234`) são enviados. Caracteres ambíguos (O/0, I/1) **não** são corrigidos; leituras fora do formato são descartadas e contadas em `invalid_plates`.
- **Erro de câmera, detector ou OCR:** o worker para e `/health/ready` passa a responder 503.
- **Privacidade:** não se loga credencial RTSP, frame completo nem placa completa por padrão.
