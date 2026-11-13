# Contexto do projeto: Microserviço ALPR (reconhecimento de placas)

> Arquivo de handoff para outra conversa do Claude. Descreve o que já foi construído, as decisões tomadas e o que falta. Baseado no estado dos arquivos do repositório até 06/11/2026 (seção 8 resume o que mudou nesta data). Os testes citados existem; a contagem de testes passando está na seção 8 e foi reportada pela execução do usuário/agente, não reverificada aqui.

## 1. Objetivo e fronteira

Este repositório contém **somente o microserviço Python de visão computacional** de um sistema de estacionamento. Ele lê um vídeo/câmera, detecta a placa, lê os caracteres, aplica qualidade e deduplicação e (futuramente) envia o evento a um backend Node.js/Express de **outra equipe**.

Não pertence a este repo: CRUD de visitantes, banco/ORM, regras de reserva/autorização, implementação do backend, nem inventar o contrato (rota, payload) do backend.

Stack declarada: Python 3.11, FastAPI, Uvicorn, Pydantic, pydantic-settings, HTTPX, OpenCV, NumPy, TensorFlow, Pytest.

## 2. Método de trabalho (skills.md)

O repo tem um `skills.md` (guia para assistentes de IA). Pontos principais:
- Trabalhar em fatias pequenas e testáveis, explicando o código no ponto de uso (o aluno quer entender, não só receber tudo pronto).
- Inspecionar o repo antes de propor mudanças; nunca afirmar que rodou/testou algo sem ter rodado.
- Perguntar no máximo uma coisa objetiva se houver bloqueio; senão, assumir de forma reversível e documentar.
- Não converter incógnitas em fatos (modelo de detecção/OCR, contrato do backend, limiares).
- Valores de limiares são provisórios até serem medidos com dados reais.
- Marcos sugeridos: 1 fundações, 2 fonte de frames, 3 contrato de resultado, 4 detecção e OCR, 5 pipeline, 6 deduplicação, 7 backend, 8 API/worker, 9 validação ponta a ponta.
- Resposta preferida: Objetivo, Código/design atual, Passos, Explicação, Verificação, Próximo passo.

## 3. Estado atual do código

### Concluído (código + testes unitários existentes)

**`app/config.py`**: `Settings` (pydantic-settings, lê `.env`, `extra="ignore"`) com `camera_source="data/teste.mp4"`, `camera_id="portaria-1"`, `backend_url="http://localhost:3000"`, `backend_api_key=""`, `min_confidence=0.6` (0..1), `dedup_seconds=30` (>0), `sample_interval_seconds=1.0` (>0). `get_settings()` com `lru_cache`. Valores são padrões de desenvolvimento, não validados. Testes: `tests/unit/test_config.py`.

**`app/camera/source.py`**: Milestone 2.
- Exceções: `CameraError`, `CameraOpenError`, `CameraReadError`.
- `parse_source`: só texto formado por dígitos vira índice de webcam; vazio levanta `ValueError`; caminho/URL ficam texto.
- `describe_source`: descrição segura para log, sem usuário/senha de URL.
- `compute_stride(fps, interval_s)`: `max(1, round(fps*interval))`; FPS inválido (0, negativo, NaN) cai em `DEFAULT_FPS=30` com warning. Amostragem por contagem de frames.
- `CameraSource(source, sample_interval_s, capture_factory=cv2.VideoCapture)`: `open()`, `frames()` (gerador, libera no `finally`), `stop()` (via `threading.Event`), `close()` idempotente, `is_live`, `fps`. Fim de arquivo termina normalmente; fonte ao vivo que para de entregar levanta `CameraReadError`.
- Testes: `test_camera_funcoes.py`, `test_camera_source.py` (com `FakeCapture`), `tests/integration/test_camera_source_video.py` (gera vídeo MJPG real).

**`app/schemas/recognition.py`**: Milestone 3.
- `BoundingBox` (frozen; x,y >= 0; width,height > 0; pixels absolutos, (x,y) = canto superior esquerdo).
- `PlateObservation` (frozen): `camera_id`, `plate_text` (como o OCR leu, sem normalizar), `detection_confidence`, `ocr_confidence` (0..1), `captured_at` (`AwareDatetime`, sem default), `bbox` opcional. Ainda **não** é o payload do backend. Testes: `test_schemas_recognition.py`.

**`app/recognition/base.py`**: `PlateDetection(bbox, confidence)`, `PlateReading(text, confidence)`, Protocols `PlateDetector.detect(frame)` e `PlateReader.read(crop) -> PlateReading | None`.

**`app/recognition/fakes.py`**: `FakePlateDetector` e `FakePlateReader` (este guarda os recortes recebidos em `.crops`).

**`app/recognition/preprocessing.py`**: `crop_region(frame, bbox)` recorta cortando o que passa da borda; devolve `None` se a caixa está totalmente fora. Testes: `test_preprocessing.py`.

**`app/services/deduplicator.py`**: Milestone 6 (política inicial).
- `Deduplicator(window_s, clock=time.monotonic)`, chave `(camera_id, plate)`.
- Janela deslizante: cada nova visão renova o prazo; no instante exato `window_s` já é nova; entradas expiradas são removidas (memória limitada); `window_s <= 0` levanta `ValueError`.
- Limitações documentadas: estado só em memória, perdido ao reiniciar, não compartilhado entre processos.
- Testes: `test_deduplicator.py` (relógio falso).

**`app/services/pipeline.py`**: Milestone 5.
- `RecognitionPipeline(detector, reader, camera_id, min_confidence, deduplicator=None)`.
- `process_frame(frame, captured_at)`: ordena detecções por confiança desc; descarta abaixo de `min_confidence`; recorta (descarta se fora do frame); lê; descarta leitura `None`/vazia/só espaços ou abaixo do limiar; monta `PlateObservation`. Falha do detector **propaga** (não vira "sem placa").
- `process_stream(frames)`: `captured_at = datetime.now(UTC)` tomado ao chegar o frame (antes da inferência); aplica o deduplicador se houver.
- O mesmo `min_confidence` vale para detecção e OCR (provisório).
- Testes: `test_pipeline.py` (inclui limiar abaixo/igual/acima e deduplicação em stream).

**`training/rds.py`**: leitor do dataset **RodoSol-ALPR** (imagens `.jpg` + `.txt` com `type`, `plate`, `layout`, `corners`).
- `parse_annotation` (4 cantos -> `BoundingBox` envolvente; erros claros), `PlateAnnotation`, `Sample`, `find_images` (rglob ordenado), `load_sample` (erro se faltar ou for inválido, citando o arquivo).
- Testes: `test_rds_parser.py`, `test_rds_loader.py`.

**Experimentos (`experiments/`)**: `ler_imagem.py` (conceitos de shape/BGR/resize), `testar_fonte.py` (lê vídeo e salva frames em `experiments/frames`), `testar_pipeline.py` (pipeline com detector falso e `CAIXA` chutada, salva recortes), `varrer_dataset.py` (conta layouts/tipos/erros do RodoSol em `data/rodosol`), `medir_recortes.py` (estatísticas de largura/altura das placas por pasta e salva alguns recortes em `experiments/crops`).

**`app/schemas/events.py`** (06/11): `PlateEvent` = evento enviado ao backend. Campos em snake_case no Python e camelCase no JSON (`alias_generator=to_camel`, `populate_by_name`). `from_observation(obs, event_id=None)` gera `event_id` (`uuid4().hex`, 32 caracteres) e normaliza a placa só para o que o backend aceita (maiúsculas, apenas A-Z e 0-9; vazio levanta `ValidationError`). `to_payload()` usa `model_dump(by_alias=True, mode="json")`. Não leva `bbox`. Isso **não** é a validação Mercosul/antiga (continua pendente).

**`app/services/backend_client.py`** (06/11): `BackendClient(base_url, api_key, timeout, max_retries, backoff_s, transport, sleep)` com um `httpx.Client` reutilizado e header `X-API-Key`. `send(event) -> bool` (True se o backend respondeu `duplicado`). Sucesso: 200/201. Repete (mesmo `eventId`, backoff exponencial 0.5s, 1s, 2s) em erro de transporte/timeout, 409, 429 e 5xx exceto 503. Não repete em 400, 401, 503 (`BackendRejected`). Esgotadas as tentativas: `BackendUnavailable`. Chave vazia levanta `ValueError`. `Retry-After` do 429 não é tratado. Testes: `tests/unit/test_backend_client.py` (`httpx.MockTransport`).

**`app/services/worker.py`** (06/11, criado pelo agente `alpr-implementador`): worker em thread própria que abre a câmera, consome `camera.frames()`, passa pelo pipeline, converte em `PlateEvent` e envia pelo `BackendClient`. Falha de envio é logada (sem chave nem placa completa) e **não derruba** o worker; o evento é perdido (sem fila). `CameraReadError`/`CameraOpenError` ou erro de detector/leitor param o worker e deixam `ready` falso. `stop()` chama `camera.stop()`, faz join com timeout e `camera.close()`. Expõe estado (running, ready, last_error) e contadores (frames, enviados, duplicados informados pelo backend, falhas de envio). Fim de vídeo gravado termina sem erro e `ready` fica falso. Testes: `test_worker.py`.

**`app/main.py`** (reescrito em 06/11): lifespan cria `CameraSource`, `BackendClient`, `Deduplicator`, pipeline (via `build_pipeline(settings, deduplicator)`, **ainda com detector/OCR fake**, com TODO para trocar pelos reais) e worker; sobe o worker no startup e encerra tudo no shutdown. `GET /health/live` (sempre 200) e `GET /health/ready` (200 só com worker rodando e câmera aberta; 503 com motivo). Sem `BACKEND_API_KEY`: nenhum worker sobe e `ready` devolve 503 "BACKEND_API_KEY não configurada". O mock `/reconhecer` e o `GET /` foram removidos. Testes: `test_main_health.py` (TestClient).

### Ferramental
- `iniciar.py` (Windows): cria `.venv` com `uv venv --python 3.11`, instala via `uv pip sync requirements.lock` (ou compila a partir de `requirements-dev.txt`) e sobe `uvicorn app.main:app --reload`.
- Dependências em `requirements.txt` / `requirements-tf.txt` / `requirements-dev.txt` + `requirements.lock` (gerado por `uv pip compile`). `pytest.ini`: `pythonpath = .`, `testpaths = tests`.
- `.gitignore` ignora `.venv/`, `.env`, `models/`, `data/`, `*.keras`, `*.h5`, `experiments/crops/`.
- `.env.example` com as variáveis de `Settings` (sem prefixo `VISION_`).

## 4. Decisões já tomadas
- Amostragem de frames por contagem (`round(fps*intervalo)`), sem `sleep` no processamento de arquivo.
- Webcam só se o texto for puramente numérico.
- Detecção e OCR separados atrás de Protocols, com fakes para teste sem modelo e sem câmera.
- Observação interna (`PlateObservation`) separada do evento do backend (ainda não definido).
- `plate_text` guardado cru; normalização ainda não existe.
- Deduplicação por (câmera, placa), janela deslizante, em memória.
- Timestamps com fuso (UTC); `captured_at` = chegada do frame ao pipeline.

- (06/11) Fonte de vídeo é **câmera USB** (`CAMERA_SOURCE=0`); sem RTSP, sem reconexão de rede. Vídeo gravado só para testes/desenvolvimento.
- (06/11) Evento de saída (`PlateEvent`) separado de `PlateObservation`; `bbox` não é enviado (o backend rejeita campo extra com 400).
- (06/11) Idempotência por `event_id` gerado no microsserviço; reenvios reutilizam o mesmo id.
- (06/11) Política de perda: falha definitiva de envio descarta o evento (sem fila/banco), registrando log sem dados sensíveis.
- (06/11) Dataset RodoSol (+9 GB) e modelos ficam fora do git e **apenas no computador do Senac**; treino, medições e avaliação com imagens reais são feitos lá (ou aqui depois que o usuário baixar os arquivos). Testes automatizados nunca podem depender dessas imagens.

## 5. Desvios e pontos de atenção observados no repo
- O `skills.md` manda usar `pyproject.toml` + `uv.lock`; o repo usa `requirements*.txt` + `requirements.lock` com `uv pip`. Não há `pyproject.toml` nos arquivos vistos. Decidir se isso é intencional.
- Nomes de variáveis de ambiente sem prefixo `VISION_` (o guia sugere, mas manda manter nomes existentes).
- `training/rds.py` tem `from anyio import Path` seguido de `from pathlib import Path`; o primeiro import é desnecessário e deve ser removido.
- `numpy` aparece apenas em `requirements-tf.txt`; `requirements.txt` não o lista, embora o código de câmera/pipeline dependa de NumPy.
- Em `tests/unit/test_pipeline.py`, `RelogioFalso` é definido no meio do arquivo (funciona, mas fica melhor junto dos helpers).
- Imports de `.env` carregam `data/teste.mp4` por padrão; `data/` está no `.gitignore`, então o vídeo de teste não vem com o repo.
- `Deduplicator` é aplicado em `process_stream`, não em `process_frame`.

## 6. O que falta (ordem sugerida)
Já feitos em 06/11: contrato com o backend, evento de saída, `backend_client.py`, worker, lifespan e health (ver seção 8).

1. **Milestone 4, treino e medição no Senac**: o código já existe (`tf_models.py`, `training/train_detector.py`, `train_ocr.py`, `evaluate.py`) e está ligado ao `build_pipeline` por `RECOGNITION_MODE=tf`. Falta só rodar o roteiro de [docs/senac.md](docs/senac.md) com o dataset, trazer `models/` (fora do git) com as fichas e `avaliacao.json`, e registrar as métricas aqui. A nota antiga de que não havia código TensorFlow estava desatualizada.
2. Normalização e validação de formato de placa (Mercosul e formato antigo brasileiro), sem "corrigir" caracteres ambíguos à força; guardar texto cru para diagnóstico. A limpeza atual em `PlateEvent.from_observation` é só para não tomar 400 do backend.
3. Limiares separados para detecção e OCR, escolhidos com dados reais; avaliação repetível em imagens rotuladas (no Senac).
4. Tratamento de falha da câmera USB: desconexão já para o worker e deixa `ready` falso; falta decidir se haverá reabertura com backoff curto (só se necessário) e validar com a webcam real (talvez `cv2.CAP_DSHOW` via `capture_factory`; registrar resolução usada).
5. Opcional no cliente HTTP: respeitar `Retry-After` do 429; avaliar o limite do backend (300 req/15 min por IP) ao medir o volume real de eventos.
6. README, `.env.example` atualizado (incluir `BACKEND_TIMEOUT` e `BACKEND_MAX_RETRIES`, conferir se já foram adicionados), documentação do contrato e da política de perda de eventos.
7. **Milestone 9**: teste ponta a ponta com clipe curto aprovado (poucos MB, versionável ou guardado fora do git) e depois câmera real; medir latências por etapa.
8. Limpezas pendentes do repo (seção 5): import `anyio.Path` em `training/rds.py`, `numpy` em `requirements.txt`, decisão `pyproject.toml` vs `requirements*.txt`.
9. Backend/Front (outra equipe, só com autorização do usuário): testes da rota `/eventos-alpr`, `.env.example` no Back_park; o Front_Park não referencia `eventos-alpr`.

## 7. Regras para continuar
- Ler os arquivos atuais antes de editar; o repo é a fonte da verdade.
- Uma fatia por vez, com teste e comando de verificação (`pytest tests/unit -q` ou similar; confirmar o comando real do projeto).
- Não adicionar fila, banco, Redis ou outro framework sem necessidade demonstrada.
- Não logar credenciais RTSP, frames completos nem placas completas por padrão.
- Responder em português, de forma direta, explicando o código novo no ponto onde aparece.
- Usar os agentes e a skill de `.claude/` (`alpr-dev`, `alpr-implementador`, `alpr-modelo`, `alpr-integrador`). Rodar testes com `.venv\Scripts\python -m pytest ... -q` (o `pytest` não está no PATH do terminal).

## 8. Registro de 06/11/2026

**Feito hoje**
- Contrato do backend levantado (somente leitura) no Back_park pelo `alpr-integrador`; o backend **já está pronto** e não precisou mudar.
- Passo 2 (cliente HTTPX): `PlateEvent`, `BackendClient`, `Settings.backend_timeout` (padrão 5) e `Settings.backend_max_retries` (padrão 3); 10 testes em `test_backend_client.py` passaram (rodados pelo usuário).
- Passo 3 (worker, lifespan, health): `worker.py`, `main.py` reescrito, `test_worker.py`, `test_main_health.py`. O agente reportou 9 testes novos passando e suíte completa `132 passed, 1 warning` (warning do TestClient/Starlette sobre httpx).
- Nada foi commitado ainda nesta data (conferir `git status`).

**Contrato do backend (Back_park, fonte da verdade)**
- `POST {BACKEND_URL}/eventos-alpr` (sem prefixo `/api`; `routes/index.ts`, `app.ts`).
- Autenticação: header `X-API-Key` comparado a `ALPR_API_KEY` do servidor (mínimo de 32 caracteres; sha256 + `timingSafeEqual`). Sem a variável no servidor: 503. Chave errada ou ausente: 401.
- Body JSON camelCase, schema `.strict()` (campo extra = 400): `eventId` (8-64), `cameraId` (1-64), `plateText` (1-20, o backend normaliza para A-Z0-9 maiúsculo), `detectionConfidence` e `ocrConfidence` (0..1), `capturedAt` (ISO 8601 com offset; `Z` serve).
- Idempotência por `eventId` (`@unique`): novo = 201; repetido = 200 com `duplicado:true`; corrida = 409 (reenviar devolve 200).
- Outros: 400 `{errors:[{path,message}]}`, 413 acima de 16kb, 429 acima de 300 req/15 min por IP, 500 inesperado. Sem deduplicação por placa/câmera no backend (cabe ao microsserviço).

**Estado atual em uma frase**: o encanamento ponta a ponta existe (câmera -> pipeline -> evento -> backend), mas com detector/OCR fake, então nada é detectado de verdade até o Milestone 4 ser feito.

**Próximo passo imediato**: ver a seção 10 (retomada no Senac).


## 9. Registro de 09/11/2026 (itens que não dependem do dataset/Senac)

**Feito**
- Normalização/validação de placa (`plate_format.py`, já existente) agora está **ligada ao worker**: leitura fora de Mercosul/antigo não é enviada, vira `invalid_plates` (exposto em `/health/ready`) e não loga o texto. Teste novo em `test_worker.py`.
- Cliente HTTP respeita `Retry-After` do 429 (usa o maior entre ele e o backoff; teto 30s; header inválido é ignorado). 2 testes novos.
- Limpezas: removido `from anyio import Path` de `training/rds.py` (era sombreado por `pathlib.Path`); `numpy>=1.26` adicionado a `requirements.txt`. Decisão: **sem `pyproject.toml`**, o projeto usa `requirements*.txt`.
- `.env.example` já tinha `BACKEND_TIMEOUT` e `BACKEND_MAX_RETRIES` (conferido).
- `README.md` criado (rodar, variáveis, health, contrato, políticas de perda de evento).
- Suíte: `.venv\Scripts\python -m pytest tests/unit -q` -> 133 passed (+ teste de `invalid_plates` no health ainda não escrito).

**Continua pendente (depende do Senac/hardware)**: Milestone 4 (modelo real), limiares de detecção/OCR com dados reais, webcam USB real (reabertura com backoff e `CAP_DSHOW` só se necessário), Milestone 9 (ponta a ponta), medir volume real versus limite 300 req/15 min.
**Commitado** em 09/11/2026 (ver `git log`).

### Atualização (mesmo dia): preparação para o Senac
- `RECOGNITION_MODE` (`fake` padrão | `tf`), `DETECTOR_MODEL_PATH`, `OCR_MODEL_PATH`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE` em `Settings` e `.env.example`. `build_pipeline` carrega `TfPlateDetector`/`TfPlateReader` no modo `tf`; modelo ausente ou corrompido vira `startup_error` e `/health/ready` 503 (worker nem sobe). Teste novo.
- Pipeline com limiar de OCR separado (`ocr_min_confidence`, padrão = `min_confidence`). Teste novo. Item 3 de 6 fica só dependente de dados.
- `app/recognition/model_card.py`: ficha JSON ao lado do modelo (sha256, data, versões TF/Keras, params, métricas, tamanho dos conjuntos). Os dois scripts de treino gravam a ficha; o serviço loga sha256/data ao subir. Testes novos.
- `training/evaluate.py`: avalia no conjunto de teste por tipo/layout (detecção IoU>=0.5, OCR isolado, ponta a ponta, formato válido, ms por etapa) e grava `models/avaliacao.json`. Parte pura (iou, contagens) testada; a parte com modelo roda só no Senac.
- `docs/senac.md`: roteiro de instalação, treino, avaliação, retorno dos arquivos e uso.
- Bug achado e corrigido: `OCR_MIN_CONFIDENCE=` vazio no `.env.example` quebrava `Settings`; agora `env_ignore_empty=True` (teste novo).
- Suíte completa: 143 passed. O código de treino/avaliação com TensorFlow real **não foi executado** aqui (sem dataset/modelo).
- Pendência: o treino não mede métricas do detector além da perda de validação; `evaluate.py` cobre isso.


## 10. Retomada no Senac (ponto de partida)

**Estado**: todo o código que não depende do dataset está pronto, testado (143 passed) e commitado. O que **nunca foi executado** é o código com TensorFlow real: `training/train_detector.py`, `training/train_ocr.py` (com a gravação da ficha adicionada em 09/11) e `training/evaluate.py`. Pode aparecer erro na primeira rodada; corrigir com teste.

**Passo a passo** (detalhes em [docs/senac.md](docs/senac.md)):
1. `git pull`, criar `.venv`, `pip install -r requirements-dev.txt`, rodar `pytest tests/unit -q` antes de qualquer treino. Anotar versões de Python/TensorFlow/Keras.
2. Dataset em `data/rodosol/` (fora do git). Treinar: `python -m training.train_detector` e `python -m training.train_ocr` (saem em `models/` com ficha `.json`).
3. Avaliar: `python -m training.evaluate` -> `models/avaliacao.json` (por carro/moto e Brazilian/Mercosul).
4. Registrar aqui os números reais (detecção, OCR por placa e por caractere, % formato válido, ms por etapa) e as versões usadas.
5. Trazer `models/*.keras`, as fichas e `avaliacao.json` para a máquina de desenvolvimento (fora do git); conferir o sha256.

**Decisões que dependem desses números**
- `MIN_CONFIDENCE`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE` (hoje 0.6 / 0.5 / vazio, provisórios).
- Se `e2e_formato_valido` ficar baixo em algum grupo (moto de duas linhas, Mercosul): decidir se treina mais, ajusta o recorte/pré-processamento ou aceita a perda. **Não** corrigir caracteres ambíguos à força.
- O detector foi treinado com frames 1280x720 reduzidos a 384 de largura; a webcam real pode ter outra resolução e proporção.

**Depois do Senac** (nesta ordem): trocar `RECOGNITION_MODE=tf` e rodar com um clipe curto aprovado (Milestone 9); webcam USB real (reabertura com backoff e `cv2.CAP_DSHOW` só se necessário; registrar a resolução); medir volume de eventos contra o limite de 300 req/15 min do backend; medir latência por etapa.

**Pendências de outras equipes (só com autorização)**: testes da rota `/eventos-alpr` e `.env.example` no Back_park; o Front_Park não referencia `eventos-alpr`.

**Commits**: sem `Co-Authored-By` (pedido do usuário); autor é o usuário (`dasilva-thiago`).


## 11. Backend: integração ALPR restaurada (09/11/2026, branch, ainda sem merge)
- Diagnóstico: os merges de 06/11 do Back_park (commit `c8c33ba`, Alice-Oliveira-22, e conflitos em `544b2b6`) removeram a rota `/eventos-alpr` de `routes/index.ts`, tiraram `ALPR_API_KEY` do `env.ts` e deixaram marcadores de conflito em `contract.prisma` (`contract.json`/`.d.ts` sem `EventoAlpr`). Resultado: POST dava 404 e o build quebrava.
- Correção no Back_park, branch `fix/integracao-alpr` (commit `a54ed8d`, **pushada, sem PR nem merge na main**): conflitos resolvidos (EventoAlpr e Movimentacao, mantendo índices/check), `npm run contract:emit` ok (storageHash igual ao da pasta `migrations/snapshots`), `ALPR_API_KEY` opcional com mínimo 32 em `env.ts`, rota e middleware religados. `npm run typecheck` passa.
- Verificado com o servidor real (porta 3099): sem chave 401, chave errada 401, corpo inválido com chave correta 400 com os erros do zod. **Não testado**: POST válido 201/200 (grava no banco de `DATABASE_URL`) e se a tabela `evento_alpr` existe no banco. Fazer com autorização do usuário.
- Pendências: abrir PR da branch e avisar a equipe (Alice) para não reverter de novo; rodar `npm test` no Back_park; criar teste da rota `/eventos-alpr`; `.env.example` no Back_park com `ALPR_API_KEY`. Front_Park: nada a fazer.

### Verificação completa do backend (09/11/2026, commit `20c906f` na branch `fix/integracao-alpr`)
- **Tabela `evento_alpr` existe** no banco (Neon, sa-east-1, o mesmo do `.env`).
- **Bug real achado pelo POST válido**: dava 500 porque a coluna TIMESTAMP exige `Temporal.PlainDateTime`, e o service passava `Date`. Corrigido com `toPlainDateTimeUtc` em `src/lib/temporal.ts` (grava em UTC).
- **Teste novo** `Back_park/tests/evento-alpr.test.mjs`: 401 sem chave/chave errada, 400 (corpo inválido, campo extra, confiança > 1, data fora do ISO), 201, 200 duplicado com o mesmo id, placa normalizada `abc-1d23` -> `ABC1D23`. Grava no banco real com `eventId` `teste-alpr-*` e apaga tudo no final (conferido: nenhuma sobra). 4/4 passam.
- `Back_park/.env.example` criado com `ALPR_API_KEY`; valor vazio é aceito (equivale a não configurada, 503).
- **`npm test` completo**: o `rbac.test.mjs` já falhava por motivo não relacionado: o banco tem os tipos `Administrador`/`Funcionario`/`Vigilante` e o teste procura `ADMIN`/`VIGILANTE`/`FUNCIONARIO`. O conserto provável é `npm run rbac:init`, que **escreve no banco compartilhado**; não rodei sem autorização. Abrir PR e combinar com a equipe.
- Falta ainda: PR da branch para a main do Back_park e um teste ponta a ponta do py_vision_park contra o backend real (com `BACKEND_API_KEY` = `ALPR_API_KEY`).

### Pendências da seção 11, resolução (09/11/2026)
- **Teste ponta a ponta feito**: `tests/integration/test_backend_real.py` (só roda com `E2E_BACKEND_URL` e `E2E_BACKEND_API_KEY`). Contra o Back_park real (porta 3099 com o banco do `.env`): `BackendClient` criou o evento (201), reenviou e recebeu `duplicado` (200), e chave errada foi rejeitada sem retry. 2 passed. Conferi no banco: placa `abc-1d23` virou `ABC1D23`, câmera e confianças corretas, `capturedAt` em UTC. Apaguei o evento de teste (tabela voltou a 0 linhas).
- **rbac.test.mjs, só em parte**: o teste agora acha os tipos por `normalizeRole` (aceita `Administrador`/`Vigilante`), e subiu de 12 falhas para só 1 subteste: `PATCH /admin/users/:id/role` devolve 503 `Catálogo RBAC não inicializado`, porque a rota busca o tipo pelo nome exato `VIGILANTE`. O banco só tem `Administrador`, `Funcionario`, `Vigilante`.
- **Decisão pendente do usuário**: `npm run rbac:init` resolveria, mas cria os tipos canônicos e **move os 35 usuários reais** (31 Funcionario, 2 Administrador, 2 Vigilante) para eles, no banco compartilhado. Foi bloqueado como mudança em recurso compartilhado e **não foi executado**. Alternativa sem tocar nos dados: a rota `/admin/users/:id/role` procurar o tipo com `normalizeRole` em vez de nome exato (mudança de código no Back_park).

### rbac:init executado (09/11/2026) - rbac.test.mjs fechado
- Com autorização do usuário, rodei `npm run rbac:init` no Back_park (banco do `.env`). Backup antes: lista de tipos e de usuário->tipo em arquivo local temporário (35 usuários, 3 tipos; não versionado).
- Resultado: criados `ADMIN` (id 7), `VIGILANTE` (8) e `FUNCIONARIO` (9). Usuários movidos: 2, 2 e 31 (total 35 conferido). Os tipos antigos (`Administrador` 2, `Funcionario` 1, `Vigilante` 3) continuam na tabela, **sem usuários**; podem ser apagados depois. Para desfazer: mover os usuários de volta para os ids 2, 1 e 3.
- `npm test` do Back_park: 13/13 passam; nenhum usuário de teste sobrou.
- Pendências restantes do ALPR: treino/avaliação no Senac; depois webcam e latências.


## 12. Registro de 10/11/2026 (Senac): treino iniciado, restante para 11/11

**Ambiente**: Python 3.11.17, TensorFlow 2.21.0, Keras 3.15.1, CPU (sem GPU), 8 núcleos. Dataset em `data/rodosol/` completo: 4 pastas x 5000 imagens (cars-br, cars-me, motorcycles-br, motorcycles-me). `pytest tests/unit -q`: **141 passed** (a seção 9 citava 143; diferença não investigada).

**Feito hoje**
- Modelos antigos de 05/11 (sem ficha `.json`, origem de um treino exploratório) movidos para `models/anteriores_05-11/` (fora do git). Não usar como resultado oficial.
- Detector: `python -m training.train_detector --per-folder 600 --epochs 12`, log em `logs/train_detector.log` (ignorado pelo git). Cerca de 100 s por época. Resultado: ver "Resultado do detector" abaixo.

**Falta (fazer em 11/11, nesta ordem, no mesmo computador do Senac)**
1. Se o detector não gravou `models/detector.keras` + ficha `.json`, repetir o passo (ver log).
2. OCR: `.venv\Scripts\python -m training.train_ocr --per-folder 1500 --epochs 15 > logs\train_ocr.log` (tempo ainda desconhecido; esperar pelo menos o equivalente ao detector, ~20 min ou mais). Para um teste rápido antes: `--per-folder 400 --epochs 8`.
3. Avaliar: `.venv\Scripts\python -m training.evaluate` (ou `--limit 500` primeiro) -> `models/avaliacao.json`.
4. Registrar aqui os números reais (detecção IoU>=0.5, OCR por placa e por caractere, % formato válido, ms por etapa; por carro/moto e Brazilian/Mercosul) e decidir `MIN_CONFIDENCE`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE` com base neles.
5. `RECOGNITION_MODE=tf` no `.env` e rodar com um clipe curto (Milestone 9); depois webcam USB real, volume de eventos vs limite de 300 req/15 min, latência por etapa.
6. Backend: o `BACKEND_API_KEY` do `.env` do py_vision_park deve ser igual ao `ALPR_API_KEY` do Back_park.

**Atenção**: o código de treino/avaliação com TensorFlow real é rodado pela primeira vez nesta data; erros na primeira execução do OCR ou do `evaluate` são esperados e devem ser corrigidos com teste.

### Resultado do detector (10/11/2026, concluído às 21:46)
- Comando: `python -m training.train_detector --per-folder 600 --epochs 12`, 12 épocas, ~100 s cada (~20 min no total, CPU). Saída: `models/detector.keras` + `models/detector.json` (fora do git).
- sha256: `57be8512298ca18b755e2eb53213442db8beb13419e84dcc82f8ddb2e0076673`. Conjuntos: 4800 treino / 200 validação. Largura de entrada 384.
- Perda final: treino 0.0105, validação 0.0090 (a menor validação vista foi 0.0079, na época 9). **Isso é só perda de treino, não é taxa de detecção**: a taxa real (IoU>=0.5) só sai do `training.evaluate`, ainda não rodado.
- Tempo de referência para o OCR: o detector levou ~20 min com 2400 imagens por época-base; o OCR (`--per-folder 1500 --epochs 15`) usa mais imagens e épocas, então reservar bem mais que isso (não medido).

### OCR rápido e avaliação rápida (10/11/2026, 21:49 a 22:00)
**Isto é só teste de funcionamento do pipeline; o OCR NÃO está bom e não deve ser apresentado como resultado.**
- OCR: `python -m training.train_ocr --per-folder 400 --epochs 8` (3200 treino / 400 validação). **9 min**, ~39 s por época. Acurácia por caractere: treino 0.299, **validação 0.152** (chance ao acaso ≈ 0.03). sha256 `9aaccac3eb53da06bd490a2baf7c7f260ae7baa3840401fc3cfd3a35f6418ca6` (`models/ocr.keras` + `models/ocr.json`). Ainda estava aprendendo (a acurácia subia a cada época); o treino foi curto demais.
- Avaliação: `python -m training.evaluate --limit 500` (125 imagens por grupo). **1,5 min**. Resultado em `models/avaliacao.json` e `logs/evaluate_rapido.log`.

| Grupo | Detecção (IoU>=0.5) | OCR por caractere | Placa inteira |
|---|---|---|---|
| car/Brazilian | 0.936 | 0.103 | 0.0 |
| car/Mercosur | 0.984 | 0.128 | 0.0 |
| motorcycle/Brazilian | 0.984 | 0.136 | 0.0 |
| motorcycle/Mercosur | 0.992 | 0.162 | 0.0 |
| **todos (500)** | **0.974** | **0.132** | **0.0** |

- Tempo por imagem (CPU): detector 47,7 ms, OCR 38,8 ms.
- **Detector: bom** (97,4% com IoU>=0.5 em 500 imagens do conjunto de teste; só 500 de 20 000, então a margem de erro é ampla).
- **OCR: ruim, esperado para 8 épocas.** Nenhuma placa inteira acertada.
- Cuidado ao ler `e2e_formato_valido` (0.974, igual à detecção): ele só diz que o texto tem a forma de uma placa (3 letras + dígitos). O OCR é um classificador por posição e aprende cedo "letra ou dígito" em cada posição, então o formato sai válido mesmo com caracteres errados. **Formato válido não significa leitura correta.** (Explicação plausível, não verificada.)

**Próximo passo (11/11)**: OCR completo `--per-folder 1500 --epochs 15`. Estimativa revisada com o tempo medido: 12 000 imagens de treino -> ~146 s por época -> **~35 a 40 min**. Depois `training.evaluate` completo (500 imagens levaram 1,5 min; o conjunto inteiro de teste, ainda não medido, deve levar bem mais). Se a acurácia do OCR continuar baixa com o treino completo, considerar mais dados/épocas ou trocar a arquitetura para CTC (decisão a tomar com os números, não antes).


## 13. Guia para a próxima sessão (escrito em 10/11/2026, 22:xx)

**Onde estamos em uma frase**: encanamento ponta a ponta pronto e testado; detector treinado e bom (97,4% de detecção em 500 imagens); **OCR ainda ruim** (modelo rápido de teste, 13% por caractere). Falta treinar o OCR completo, avaliar tudo, decidir limiares e testar com webcam.

**Ao abrir a sessão (computador do Senac, onde estão o dataset e `models/`)**
1. `cd py_vision_park`; `git status -sb` (este repo está **2 commits à frente** da origem: `987fae1`, `785c3c8`; nenhum push feito).
2. `.venv\Scripts\python -m pytest tests/unit -q` -> esperado 141 passed.
3. Conferir `models/`: `detector.keras/.json` (sha256 `57be8512...`), `ocr.keras/.json` (modelo **rápido**, sha256 `9aaccac3...`, será sobrescrito), `avaliacao.json` (da avaliação rápida), `anteriores_05-11/` (modelos antigos sem ficha; descartáveis).
4. OCR completo: `.venv\Scripts\python -m training.train_ocr --per-folder 1500 --epochs 15 > logs\train_ocr.log` (estimativa 35-40 min, ~146 s/época; confirmar na 1ª época).
5. Avaliação completa: `.venv\Scripts\python -m training.evaluate` (sem `--limit`; tempo ainda não medido, 500 imagens levaram 1,5 min).
6. Registrar números aqui (substituir a tabela "rápida" por uma completa) e decidir `MIN_CONFIDENCE`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE`.
7. Se o OCR completo ainda for fraco (por exemplo < 80% por caractere): decidir entre mais dados/épocas ou trocar para CTC. Não corrigir caracteres ambíguos à força.
8. Depois: `RECOGNITION_MODE=tf` no `.env` (ver `.env.example`; variáveis `CAMERA_SOURCE`, `BACKEND_URL`, `BACKEND_API_KEY`, `DETECTOR_MODEL_PATH`, `OCR_MODEL_PATH`), clipe curto, webcam USB (`CAMERA_SOURCE=0`; `cv2.CAP_DSHOW` e reabertura com backoff só se necessário; registrar resolução), volume vs limite de 300 req/15 min do backend, latência por etapa (referência: detector 48 ms + OCR 39 ms por imagem em CPU).

**Backend (Back_park, outra equipe)**
- `POST /eventos-alpr` com `X-API-Key` existe e foi verificado em 09/11 (seção 11). O `BACKEND_API_KEY` do py_vision_park deve ser **igual** ao `ALPR_API_KEY` do Back_park (mínimo 32 caracteres; nunca commitar nem colar em chat).
- Em 10/11 o Back_park local estava em `main`, **1 commit atrás** da origem (`git pull` antes de testar). A tabela `evento_alpr` existe no Neon.
- Aviso à equipe (resolvido em 05/11): login/cadastro davam 500 por `contract.json` desatualizado (`usuario.status` inexistente no banco). Regra: após mudar `contract.prisma` ou rodar `contract infer`, rodar `npx prisma contract emit`, reiniciar e commitar os três arquivos `contract.*` juntos. Quem não conseguir rodar o servidor: conferir `.env` (copiar de `.env.example`), `git pull`, `npm install`.
- Diferença conhecida entre o banco e o contrato antigo: tabela `movimentacao` existe; `visitante.telefone` e `veiculo_visitante.placa` sem unicidade; `reserva_vaga.placa_reserva` aceita nulo. Alinhar com a equipe.
- Não editar backend/frontend sem autorização do usuário.

**Apresentação do PI (Senac)**: versão curta e formatada para celular foi entregue no chat (OpenCV captura/prepara a imagem; TensorFlow detecta a placa e faz o OCR; Python valida e envia por HTTPX). Atualizar com os números reais depois da avaliação completa; **não apresentar números do OCR rápido como resultado**. Dataset: 20 000 imagens (4 pastas x 5 000: cars-br, cars-me, motorcycles-br, motorcycles-me).

**Regras de trabalho do usuário (manter)**: explicar conceitos novos de forma curta; marcar `DEPENDE DO BACKEND` e `SUPOSIÇÃO` quando couber; commits e push só com as credenciais do usuário (`dasilva-thiago`), **sem Co-Authored-By**; nunca afirmar que testou o que não rodou; modelos e dataset ficam fora do git.


## 14. Registro de 12/11/2026: OCR completo e avaliação completa (Senac)

**Treino do OCR**: `python -m training.train_ocr --per-folder 1500 --epochs 15` (12 000 treino / 400 validação), 19:09 a 19:49, **~40 min** (144 s por época), confirmando a estimativa. Acurácia por caractere na validação: 0.13 -> 0.81 (época 15), ainda subindo devagar (0.77 na época 10, 0.81 na 15). sha256 `2c7e6c51d94147a398f63a0253d37ad2897cfbd9cf33ea3eb77fd71cf19e96f9` (`models/ocr.keras` + `ocr.json`). O OCR rápido de 10/11 foi guardado em `models/anteriores_05-11/ocr_rapido.*`.

**Avaliação completa**: `python -m training.evaluate` em **4 000 imagens** (1 000 por grupo), ~10 min (19:50 a 20:00). Arquivo: `models/avaliacao.json`, log `logs/evaluate.log` (a avaliação rápida de 10/11 está em `models/anteriores_05-11/avaliacao_rapida.json`).

| Grupo | n | Detecção | OCR por caractere | OCR placa inteira | Ponta a ponta (placa inteira) |
|---|---|---|---|---|---|
| car/Brazilian | 1000 | 0.926 | 0.759 | 0.203 | 0.149 |
| car/Mercosur | 1000 | 0.986 | 0.837 | 0.337 | 0.278 |
| motorcycle/Brazilian | 1000 | 0.986 | 0.855 | 0.379 | 0.361 |
| motorcycle/Mercosur | 1000 | 0.967 | 0.846 | 0.293 | 0.229 |
| **todos** | **4000** | **0.966** | **0.824** | **0.303** | **0.254** |

- Tempo médio por imagem (CPU): detector 50,6 ms, OCR 41,2 ms (~92 ms no total, ~11 imagens/s no máximo).
- **Leitura**: o detector está bom (96,6%; o pior grupo é carro brasileiro, 92,6%). O **OCR ainda não é bom o bastante**: 82% por caractere, mas só **30% da placa inteira** acertada (e **25% ponta a ponta**). Isso é coerente com a conta: 7 posições com ~82% cada dão ~0.82^7 ≈ 25% de placa inteira. Acertar a placa inteira exige acurácia por caractere bem mais alta (95% por caractere daria ~70%).
- `e2e_formato_valido` (0.966) continua igual à detecção e **não mede leitura correta** (ver seção 12). Não usar como métrica de qualidade.
- Em produção, com webcam, a condição será mais difícil que o dataset (resolução, ângulo, luz); esperar números piores.

**Decisões pendentes (dependem desses números)**
1. **Melhorar o OCR** antes de apresentar como "lê placas": mais dados (`--per-folder` maior, até 5000 por pasta = 20 000), mais épocas (a curva ainda subia), e/ou trocar a arquitetura para CTC. Ordem sugerida: primeiro dados + épocas (mesma arquitetura, ~1 h+), medir de novo; só trocar para CTC se não chegar perto de 95% por caractere. Reservar o `evaluate` completo depois (~10 min).
2. **Limiares** (`MIN_CONFIDENCE`, `DETECTOR_THRESHOLD`, `OCR_MIN_CONFIDENCE`): ainda provisórios (0.6 / 0.5 / vazio). O `evaluate` não mede acerto por faixa de confiança; para escolher o corte com dados, é preciso estender o `evaluate` para gravar a confiança de cada leitura e se estava certa (curva acerto x cobertura). Não escolher no chute.
3. Como a política do serviço é enviar só placas com formato válido e confiança suficiente, vale medir quantas leituras **erradas** passam pelo filtro (placas erradas gravadas no banco são pior que placas não lidas).


## 15. Diagnóstico do OCR (12/11/2026, 20:08; `experiments/diagnostico_ocr.py`, ~1 min)
Lê o recorte do **gabarito** (só o OCR), 800 imagens de treino e 800 de teste, sem augmentação. Saída em `logs/diagnostico_ocr.json`; mosaico do que o OCR enxerga em `logs/mosaico_ocr.png`.

**Medido**
- **Treino 88,0% por caractere (50% placa inteira) x teste 82,4% (31,6%)**. A diferença é pequena: o modelo **ainda não aprendeu nem o treino** (subajuste). Mais treino/capacidade deve subir os dois; mais dados sozinho ajuda menos.
- O split tem 12 000 treino / 4 000 val / 4 000 teste; o treino usou só **6 000** (`--per-folder 1500`) x 2 recortes. Há mais 6 000 disponíveis.
- **Erros não são de tipo** (letra no lugar de dígito: 3 de 987). São **confusões entre formas parecidas**: 4/1, 7/1, L/M, H/M, K/M, N/M, D/O, Q/O, I/J, E/F.
- **Posição 3 (índice 2) é a pior: 53,5% no teste** (69% no treino); as demais ficam entre 79% e 95%. Causa **não confirmada** (hipótese: caractere ao lado do hífen nas placas brasileiras e corte do recorte de moto).
- Placas por nº de erros (teste): 0 erros 253, **1 erro 275**, 2 erros 157, 3+ 115 (de 800). Um terço das placas erra só 1 caractere.
- Confiança mínima média: placa certa 0,49 x placa errada 0,33 (há separação, mas fraca; não escolher limiar sem curva acerto x cobertura).
- Por grupo (caractere, teste): car/Brazilian 76,6%, car/Mercosur 85,1%, moto/Brazilian 85,3%, moto/Mercosur 82,5%.

**Observado nas imagens**: os recortes são **muito borrados** (placa pequena no frame; letras como E/F e 1/4/7 ficam quase indistinguíveis a olho). Em moto, o pré-processamento junta as duas linhas lado a lado e parte das letras da linha de cima aparece cortada. Há um teto de qualidade imposto pela resolução da fonte.

**Estimativa (NÃO medida) para treino maior** (12 000 imagens, ~35 épocas, mesma arquitetura): por caractere teste ~88-90%, placa inteira isolada ~40-50%, ponta a ponta ~35-45%. Chegar a 95% por caractere (placa inteira ~70%) provavelmente exige mudar entrada/arquitetura (mais resolução, mais capacidade) e não é garantido pela qualidade das imagens. Placa inteira ≈ (acerto por caractere)^7, com um pouco de folga pela correlação dos erros.

**Mitigação de maior alavanca (hipótese a medir): votação entre vários frames.** Um carro aparece em vários frames da webcam; combinar as leituras (voto por posição ou maior confiança) tende a corrigir erros aleatórios de borrão. **Atenção ao desenho atual**: a deduplicação deixa passar a **primeira** leitura e bloqueia o mesmo (câmera, placa) por 30 s; se a primeira leitura estiver errada, vira a placa errada gravada. Como a leitura errada gera outra "placa" diferente, ela também passa; o resultado é eventos duplicados e errados. Revisar isso (agregar leituras por veículo antes de enviar) é decisão de projeto pendente.


## 16. Diagnóstico aprofundado e decisão (12/11/2026, Opus 5.5)

**Medido**
- **Viés regional**: posições 1 e 2 têm poucas letras dominantes (pos1: O/Q/M = 72%; pos2: R/P/D = 63%; placas do ES). A pos3 é uniforme (26 letras, 4,57 bits). Acerto por posição no teste (recorte do gabarito): pos3 = **0.48 / 0.62 / 0.57 / 0.46** (car-BR / car-ME / moto-BR / moto-ME). Ou seja, a **capacidade real de ler letras é ~55%**; pos1/pos2 parecem boas porque o modelo decorou o viés. Em outra região (fora do ES) pos1/pos2 devem cair.
- **Resolução não é o gargalo principal**: placa de carro tem ~125x42 px no original (p10 95x32), quase o tamanho da entrada do OCR (128x48). Moto ~103x89.
- **Ruído de rótulo no treino**: o palpite carro/moto por proporção (`CAR_MIN_ASPECT=2.0`) acerta 100% com a caixa do gabarito, mas erra **~5% dos carros** com a caixa tremida usada no treino (`jitter_box`), que viram "moto" e são cortados ao meio com o rótulo certo. Proporção: carro p5 2,53; moto p95 1,33.
- O treino salva a **última** época, não a melhor; não há parada antecipada.

**Referência externa** (não medido por nós): no artigo original do RodoSol-ALPR (Laroca et al., 2022, arXiv 2201.00267), 12 modelos de OCR tiveram **média ~50,8%** de reconhecimento neste dataset (divisão tradicional; números por modelo não conferidos). Um artigo de 2026 (arXiv 2601.07671) reporta OpenALPR 69,3% e Sighthound 57,0% no RodoSol. O dataset é difícil de propósito. **Alvo realista: 50-70% de placa inteira**, não 95%.

**Decisão (ordem)**
1. Correções baratas: no treino usar o tipo de veículo do gabarito (não o palpite); `CAR_MIN_ASPECT` revisto com dados para a inferência; parada antecipada e salvar a melhor época (val_loss).
2. Medir honesto: `evaluate` passa a mostrar acerto por posição (pos3 = leitura real de letra).
3. Trocar o OCR por **CRNN com CTC** (convolucional, sem LSTM para caber em CPU), mantendo o atual como `--arch positions` para comparação. Motivo: o atual (Flatten + Dense, 7 saídas fixas) precisa decorar onde cada caractere fica, e carro/moto têm posições diferentes; CTC lê da esquerda para a direita sem posição fixa e é o padrão para texto em imagem.
4. Teste rápido do CTC (400/pasta, 8 épocas, ~10 min) para garantir que aprende; depois treino com todas as 12 000 imagens de treino e parada antecipada; `evaluate` completo e comparação com a seção 14.
5. Adiado: votação entre frames (só mede com vídeo/webcam; o RodoSol tem uma imagem por veículo) e modelo pré-treinado de terceiros (ex.: fast-plate-ocr; licença e cobertura de placas brasileiras **não verificadas**).

### Resultado do CTC (12/11/2026, 20:28 a 21:04) - descartado por ora
- Implementado por agente Sonnet (commit `029b926`): `--arch {positions,ctc}`, `--copies`, parada antecipada (val_loss, paciência 4, melhor época), tipo de veículo do gabarito no treino, `decode_ctc`, `TfPlateReader` reconhece os dois formatos, `ocr_por_posicao`/`e2e_por_posicao` no `evaluate`. 151 testes passaram.
- Treino `--arch ctc --per-folder 3000 --copies 2` (24 000 recortes), ~200 s/época. Validação por caractere: 0.10, 0.30, 0.48, 0.53, 0.53, 0.57, 0.57, 0.57, **0.59** (época 9); placa inteira só **5%**. O positions antigo tinha ~0.70 na época 7 e terminou em 0.81.
- **Parado manualmente na época 9** (val_loss estabilizado em ~9.37; ia perder para o positions). Nenhum modelo CTC completo foi salvo. A métrica por caractere compara posição a posição e penaliza leituras com comprimento errado, mas a placa inteira (5%) confirma o resultado.
- Hipótese (não testada): CRNN só convolucional não tem contexto suficiente entre as fatias; um CTC com BiLSTM poderia ir melhor, mas é mais lento em CPU. Fica como opção futura.
- Em seguida: `--arch positions --per-folder 3000 --copies 1 --epochs 30` (correções aplicadas, todas as 12 000 de treino) -> `models/ocr_pos_v2.keras`, log `logs/train_ocr_pos_v2.log`.


## 17. Encerramento de 12/11/2026 (21:40, computador do Senac desligado)
- **OCR positions v2**, validação por época (caractere / placa inteira): 1: 0.136/0.000; 2: 0.184/0.005; 3: 0.380/0.008; 4: 0.492/0.020; 5: 0.562/0.040; 6: 0.580/0.058; 7: 0.650/0.105; 8: 0.599/0.075; 9: 0.725/0.158; 10: 0.731/0.163; 11: 0.766/0.210; 12: 0.747/0.193; 13: 0.775/0.238; 14: **0.806/0.330** (último visto antes de desligar; val com só 400 recortes, oscila). ~2,6 min/época. Praticamente empatado com o v1 época a época. **Interrompido pelo desligamento; nenhum modelo foi salvo** (o script só salva no fim do treino).
- Conclusão: com a arquitetura atual, a projeção (não medida) é ~85-88% por caractere e ~35-45% de placa inteira, o que é **insatisfatório**. Melhorias pequenas de treino não bastam.
- **Plano definitivo em [docs/plano_ocr.md](docs/plano_ocr.md)** (para Opus 5.5 / Fable): checkpoint por época, cache e GPU (WSL2/Colab, conferindo a licença do dataset); retificação pelos 4 cantos; nova arquitetura; augmentação e placas sintéticas contra o viés regional; modelo pré-treinado como linha de base; filtro de confiança, votação entre frames e redesenho do deduplicador; critérios de sucesso fixados antes de treinar.
- **Arquivos fora do git que só existem no computador do Senac** (copiar para pendrive/nuvem se possível): `models/detector.keras` + `.json` (bom, 96,6%), `models/ocr.keras` + `.json` (v1), `models/avaliacao.json`, `models/anteriores_05-11/`, `logs/`. Se forem perdidos, o detector precisa ser retreinado (~20 min).
