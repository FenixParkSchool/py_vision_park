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
