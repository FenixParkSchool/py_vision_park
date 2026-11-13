# py_vision_park — microsserviço ALPR

Lê frames de uma câmera USB (ou de um vídeo gravado, para testes), detecta a placa, faz OCR e envia o evento
ao backend (`POST {BACKEND_URL}/eventos-alpr`). Hoje o detector e o OCR são **fakes**
por padrão (`RECOGNITION_MODE=fake`). Com `RECOGNITION_MODE=fpo` (**recomendado**) usa o nosso detector (`models/detector.keras`) + o OCR pronto fast-plate-ocr. Com `RECOGNITION_MODE=tf` usa os modelos Keras próprios de `models/` (treino e avaliação: [docs/senac.md](docs/senac.md)).

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
| `CAMERA_SOURCE` | `data/teste.mp4` | índice da webcam USB (ex.: `0`) ou caminho de vídeo para testes |
| `CAMERA_ID` | `portaria-1` | identificador enviado no evento |
| `BACKEND_URL` | `http://localhost:3000` | base do Back_park (sem `/api`) |
| `BACKEND_API_KEY` | vazio | enviada no header `X-API-Key`; obrigatória |
| `MIN_CONFIDENCE` | `0.6` | confiança mínima da detecção (provisório) |
| `DEDUP_SECONDS` | `30` | janela para não repetir a mesma placa (provisório) |
| `SAMPLE_INTERVAL_SECONDS` | `1.0` | intervalo entre frames processados (provisório) |
| `BACKEND_TIMEOUT` | `5` | timeout por requisição, em segundos |
| `BACKEND_MAX_RETRIES` | `3` | tentativas extras após falha (backoff 0.5s, 1s, 2s...) |
| `RECOGNITION_MODE` | `fake` | `fake`, `tf` (modelos Keras próprios) ou `fpo` (detector próprio + fast-plate-ocr, recomendado) |
| `FPO_MODEL` | `cct-s-v2-global-model` | modelo do fast-plate-ocr, só em `fpo`; falha ao carregar = `/health/ready` 503 |
| `DETECTOR_MODEL_PATH` / `OCR_MODEL_PATH` | `models/detector.keras` / `models/ocr.keras` | usados só em `tf`; modelo ausente = `/health/ready` 503 |
| `DETECTOR_THRESHOLD` | `0.5` | limiar do mapa de probabilidade do detector |
| `OCR_MIN_CONFIDENCE` | vazio | se vazio, usa `MIN_CONFIDENCE` também no OCR |

## OCR fast-plate-ocr (modo `fpo`)
- Pacote `fast-plate-ocr[onnx]==1.1.0` (licença **MIT**), inferência ONNX Runtime em CPU. O detector continua sendo o nosso.
- Medido no teste do RodoSol (4000 imagens): **78,9% de placa inteira ponta a ponta** (nosso OCR: 25,4%). *Teste RodoSol; pode ser otimista porque não se sabe se o RodoSol estava no treino do modelo.* Confiança = média das probabilidades dos caracteres reais (sem padding). Detalhes e curva por limiar: `models/avaliacao_fpo_integrado.json` (gerado por `experiments/avaliar_fpo.py`).
- **Pesos offline:** na 1ª execução o pacote baixa o modelo (precisa de internet) para `~/.cache/fast-plate-ocr/<modelo>/` (Windows: pasta `.cache/fast-plate-ocr` dentro do perfil do usuário; contém um `.onnx` e um `_plate_config.yaml`). Para subir sem internet: rode uma vez com internet (por exemplo `.venv\Scripts\python -c "from app.recognition.fpo_reader import FastPlateReader as R; R.from_model('cct-s-v2-global-model')"`) e copie essa pasta para o mesmo caminho na máquina de destino. Sem cache e sem internet, o serviço sobe com `/health/ready` 503.

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
- **Privacidade:** não se loga frame completo nem placa completa por padrão.
