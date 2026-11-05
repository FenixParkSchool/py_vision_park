c# Contexto do projeto: Microserviço ALPR (reconhecimento de placas)

> Arquivo de handoff para outra conversa do Claude. Descreve o que já foi construído, as decisões tomadas e o que falta. Baseado no estado dos arquivos do repositório até 05/11/2026. Não afirmo que os testes foram executados por mim; a lista abaixo diz apenas que os testes existem.

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

**`app/main.py`**: ainda é **mock**: `GET /` health simples e `POST /reconhecer` devolvendo placa fixa `ABC1D23`. Não usa câmera, pipeline nem configuração.

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

## 5. Desvios e pontos de atenção observados no repo
- O `skills.md` manda usar `pyproject.toml` + `uv.lock`; o repo usa `requirements*.txt` + `requirements.lock` com `uv pip`. Não há `pyproject.toml` nos arquivos vistos. Decidir se isso é intencional.
- Nomes de variáveis de ambiente sem prefixo `VISION_` (o guia sugere, mas manda manter nomes existentes).
- `training/rds.py` tem `from anyio import Path` seguido de `from pathlib import Path`; o primeiro import é desnecessário e deve ser removido.
- `numpy` aparece apenas em `requirements-tf.txt`; `requirements.txt` não o lista, embora o código de câmera/pipeline dependa de NumPy.
- Em `tests/unit/test_pipeline.py`, `RelogioFalso` é definido no meio do arquivo (funciona, mas fica melhor junto dos helpers).
- Imports de `.env` carregam `data/teste.mp4` por padrão; `data/` está no `.gitignore`, então o vídeo de teste não vem com o repo.
- `Deduplicator` é aplicado em `process_stream`, não em `process_frame`.

## 6. O que falta (ordem sugerida)
1. **Milestone 4, detector e OCR reais**: decidir modelo/estratégia (treinar com RodoSol-ALPR ou usar pesos aprovados), contrato de entrada/saída, cor BGR vs RGB, normalização, licença. Ainda **não existe** nenhum modelo nem código TensorFlow. Já há leitor de dataset e medições de tamanho de placa para embasar isso.
2. Normalização e validação de formato de placa (Mercosul e formato antigo brasileiro), sem "corrigir" caracteres ambíguos à força; guardar texto cru para diagnóstico.
3. Limiares separados para detecção e OCR, escolhidos com dados reais; avaliação repetível em imagens rotuladas.
4. **Milestone 7, `services/backend_client.py`** com HTTPX: depende do contrato acordado com a equipe do backend (URL, método, rota, autenticação, campos, ID do evento/idempotência, retries). Ainda indefinido; não inventar.
5. Payload/evento de saída e `event_id`, separado de `PlateObservation`.
6. **Milestone 8**: worker contínuo (thread/processo separado, nunca loop dentro de rota HTTP), `main.py` com lifespan, health `live`/`ready` refletindo o estado real, substituir o mock `/reconhecer`.
7. Reconexão com backoff para fonte ao vivo (hoje a fonte ao vivo apenas levanta `CameraReadError`).
8. README, `.env.example` atualizado, documentação do contrato e da política de perda de eventos.
9. **Milestone 9**: teste ponta a ponta com clipe aprovado e depois câmera real; medir latências por etapa.

## 7. Regras para continuar
- Ler os arquivos atuais antes de editar; o repo é a fonte da verdade.
- Uma fatia por vez, com teste e comando de verificação (`pytest tests/unit -q` ou similar; confirmar o comando real do projeto).
- Não adicionar fila, banco, Redis ou outro framework sem necessidade demonstrada.
- Não logar credenciais RTSP, frames completos nem placas completas por padrão.
- Responder em português, de forma direta, explicando o código novo no ponto onde aparece.
