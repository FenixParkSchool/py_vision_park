---
name: alpr-dev
description: Fluxo enxuto para terminar o microsserviço ALPR (Python/FastAPI/OpenCV/TensorFlow). Use ao implementar, corrigir ou testar qualquer parte deste repositório.
---

# alpr-dev: terminar o microsserviço rápido e barato

Contexto completo do projeto: `contexto_projeto.md` (ler uma vez, não reler a cada tarefa).
Esta skill tem prioridade sobre o tom didático do `skills.md`. As fronteiras do `skills.md` continuam valendo.

## Fonte de vídeo: câmera USB (decisão do usuário, prioridade alta)
O projeto será testado com uma **câmera USB (webcam)**. **Não será mais usada câmera de rede**: sem RTSP, sem IP, sem credenciais de câmera.
- A fonte é um índice numérico em `CAMERA_SOURCE` (ex.: `0`, `1`). `parse_source` já converte texto numérico em índice. Vídeo gravado continua válido só para testes automatizados e desenvolvimento.
- Não implementar nem testar nada de RTSP/URL/reconexão de rede. Remover ou ignorar essa preocupação.
- Webcams costumam reportar FPS 0 ou errado: `compute_stride` já cai em 30 FPS. Se a amostragem ficar errada na prática, medir e ajustar em vez de supor.
- Se a câmera não abrir no Windows com o backend padrão do OpenCV, testar `cv2.CAP_DSHOW` passando o flag pelo `capture_factory` (isolado, sem espalhar pelo código). Só usar se houver problema real.
- Desconectar o cabo USB gera `CameraReadError`: tratar como falha de fonte (worker para e `ready` fica falso; reabrir com backoff curto só se necessário).
- Uma câmera USB só pode ser aberta por um processo/leitor por vez: garantir um único `CameraSource` e que `release()` aconteça no desligamento.
- Resolução e exposição afetam a leitura da placa: ao validar, registrar a resolução usada e testar com a placa a distâncias reais.
- Testes automatizados continuam sem hardware (fakes). A validação com a webcam real é manual e curta (script em `experiments/`).

## Regras de economia
1. Ler só os arquivos que a tarefa toca. Nunca varrer o repo inteiro.
2. Uma fatia por vez: implementar, rodar os testes **só do que mudou** (`pytest tests/unit/test_X.py -q`), seguir em frente.
3. Suíte completa uma vez por marco concluído, não a cada edição.
4. Sem auditorias, revisões gerais ou refatorações não pedidas. Corrigir só o que bloqueia a tarefa.
5. Não reabrir decisões já tomadas (seção 4 do `contexto_projeto.md`).
6. Decidir com padrão razoável e anotar a suposição em uma linha. Perguntar ao usuário só se for bloqueio real.
7. Explicar pouco: o que mudou (arquivos), resultado dos testes realmente rodados, próximo passo. Detalhe didático só se o usuário pedir.
8. Não criar arquivos, abstrações ou dependências além do necessário.
9. Nunca afirmar que testou algo que não rodou.

## Backend e frontend (regra crítica)
O usuário tem os repositórios do backend (Node/Express) e do frontend acessíveis nesta sessão.
- **Consultar à vontade** (somente leitura) para descobrir rota, método, payload, autenticação e idempotência. Preferir isso a perguntar ao usuário.
- **Nunca editar o backend nem o frontend sem avisar.** Se o microsserviço precisar de algo que o backend não oferece (rota nova, campo novo, autenticação, idempotência por `event_id`, mudança de schema), **parar e avisar o usuário** neste formato:

  > ALTERAÇÃO NO BACKEND NECESSÁRIA
  > - O que falta: ...
  > - Onde (arquivo/rota): ...
  > - Mudança proposta (mínima): ...
  > - Impacto no microsserviço: ...

  Só alterar depois da autorização do usuário. Enquanto isso, continuar o que não depende dela.
- Nunca inventar contrato: o cliente HTTP segue o que o backend realmente expõe.

## Divisão de trabalho (agentes em `.claude/agents/`)
- `alpr-implementador`: escreve código e testes de uma fatia. Padrão para a maioria das tarefas.
- `alpr-modelo`: escolhe e integra detector e OCR. Só no marco de detecção/OCR.
- `alpr-integrador`: lê backend/frontend, define o cliente HTTP e avisa sobre alterações no backend.

O agente principal coordena, delega fatias independentes em paralelo e não repete o que delegou.

## Ordem de execução (o que falta)
1. Detector e OCR reais (`alpr-modelo`) + normalização de placa Mercosul/antiga.
2. Contrato com o backend e `services/backend_client.py` (`alpr-integrador`).
3. Worker contínuo + `main.py` com lifespan + health `live`/`ready` (`alpr-implementador`).
4. Tratamento de falha da câmera USB (desconexão -> estado de falha/`ready` falso; backoff curto opcional).
5. Validação ponta a ponta: primeiro com vídeo gravado, depois com a **câmera USB** (`CAMERA_SOURCE=0`), README e `.env.example`.

## Limites fixos
- Este repo não tem banco, ORM, visitantes, reservas nem regra de negócio do estacionamento.
- Loop da câmera nunca dentro de rota HTTP.
- Não logar frames inteiros nem placas completas por padrão.
- Sem fila, Redis ou novo framework sem necessidade demonstrada.
- Responder em português, direto.
