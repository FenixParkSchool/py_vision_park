# Plano definitivo para o OCR de placas (para Opus 5.5 / Fable)

Escrito em 12/11/2026, ao fim da sessão no Senac. Ler junto com `contexto_projeto.md`, seções 12 a 17.

> **Status em 14/11/2026:** os critérios de sucesso da seção 3 foram atingidos **no teste do RodoSol** com o ajuste fino do fast-plate-ocr `cct-s-v2` (não com o nosso OCR): **94,0%** de placa inteira ponta a ponta (alvo 70%, excelente 85%). Com limiar 0.95, 98,8% de precisão cobrindo 94,5% (alvo: 98% cobrindo 60%). OCR 14 ms em CPU (alvo 50 ms). **Falta o critério "webcam/vídeo"** e medir o viés regional fora do ES. Os passos 1 a 3 deste plano (retificação, nova arquitetura, sintéticas) ficam **suspensos**: só voltam se a validação real mostrar falha que o ajuste fino não resolva. Detalhes: `contexto_projeto.md`, seções 20 e 21.
Objetivo: sair de ~25% de placa inteira (ponta a ponta) para um resultado **excelente e mensurável** no RodoSol-ALPR e utilizável com webcam.

## 1. Onde estamos (medido no conjunto de teste do RodoSol, 4000 imagens)

| Componente | Resultado | Situação |
|---|---|---|
| Detector (`models/detector.keras`, sha256 `57be8512...`) | **96,6%** IoU>=0.5 (pior grupo: carro BR 92,6%) | bom, manter |
| OCR positions v1 (`models/ocr.keras`, sha256 `2c7e6c51...`) | 82,4% por caractere, **30,3%** placa inteira isolada, **25,4%** ponta a ponta | insatisfatório |
| OCR CTC só convolucional (parado na época 9) | 59% por caractere (val), 5% placa inteira (val) | pior que o v1, descartado |
| OCR positions v2 (correções; 12 000 imagens; parada antecipada) | val 76,6% por caractere / 21,0% placa na época 11 de 30, ainda subindo (v1 tinha 77,3% na época 11) | **interrompido pelo desligamento; NÃO foi salvo** |

Tempo por imagem em CPU: detector ~51 ms, OCR ~41 ms.

Referência externa (não medida por nós): média de ~50,8% de 12 modelos de OCR no RodoSol (Laroca et al., 2022, arXiv 2201.00267); OpenALPR 69,3% e Sighthound 57,0% (arXiv 2601.07671). Conferir os números por modelo no artigo antes de citar.

## 2. Causas identificadas (medidas; seções 15 e 16 do contexto)
1. **Modelo fraco em ler letras**: a 3ª posição (letra sem viés) acerta 46-62%; dígitos 86-97%.
2. **Viés regional**: pos1 O/Q/M = 72%, pos2 R/P/D = 63% (placas do ES). O acerto aparente nessas posições é memorização e cai fora do ES.
3. **Arquitetura**: Flatten + Dense com 7 saídas fixas precisa memorizar a posição de cada caractere; carro (1 linha) e moto (2 linhas costuradas lado a lado) têm geometrias diferentes.
4. **Recorte não retificado**: a caixa é o retângulo envolvente dos 4 cantos; placas inclinadas entram tortas e com fundo. A moto é cortada no meio da caixa, e a linha divisória real raramente fica no meio (letras cortadas no mosaico `logs/mosaico_ocr.png`).
5. **Treino sem checkpoint por época** (o commit `029b926` já trouxe tipo do gabarito, parada antecipada e melhor época; falta salvar a cada época).
6. Imagens borradas (pedágio). A resolução nativa (~125x42 px no carro) não é o gargalo principal.

## 2.1 Revisão do código de treino (`training/train_ocr.py`, 12/11, Opus 5.5)
Conclusão: **há problemas reais no código, mas eles não são a causa principal**. Corrigi-los é barato e deve dar ganho modesto (estimativa não medida: +2 a +5 pontos por caractere). Os ganhos grandes vêm da retificação, da arquitetura e dos dados (passos 1 a 4). Fazer a revisão como **passo 0b**, junto da infraestrutura.

| # | Achado | Efeito | Correção |
|---|---|---|---|
| R1 | O tremor da caixa (`jitter_box`) é feito **uma vez no carregamento**; as 30 épocas veem exatamente os mesmos recortes | augmentação fraca; estagnação | recortar/tremer **a cada época** (gerador `tf.data` sobre imagens e caixas, ou várias versões em cache sorteadas por época) |
| R2 | `random_brightness`/`random_contrast` são aplicados **depois do `.batch`**: o mesmo fator vale para as 64 imagens do lote | augmentação quase nula por imagem | aplicar por imagem (`map` antes do `batch`) |
| R3 | Sem normalização de contraste por imagem (só pixel/255) | placas escuras/borradas ficam com pouco contraste | padronização por imagem (média 0, desvio 1) ou CLAHE no `ocr_input` (treino e inferência iguais) |
| R4 | Validação com só **400** recortes (100 por pasta), tremidos | ruído de ±2-3 pontos entre épocas (vimos 0.65 -> 0.60 -> 0.73); a parada antecipada decide com base em ruído | validar com as 4000 de validação, caixa do gabarito |
| R5 | `Flatten -> Dense(256)`: gargalo pequeno e perda da noção de posição | limita a leitura de letras (pos3 46-62%) | resolvido no passo 2 (arquitetura) |
| R6 | Sem augmentação de desfoque, ruído, perspectiva ou JPEG | não treina para o pior caso real (webcam) | passo 3 |
| R7 | Moto cortada no meio da caixa envolvente | letras cortadas | passo 1 (retificação) |
| R8 | Salva só no fim | interrupção perde tudo | passo 0 (checkpoint por época) |

Não encontrei bug de rótulo, de alfabeto ou de divisão treino/teste (o `pick` usa as 12 000 de treino com `--per-folder 3000`; o split é por pasta e estável). O CTC está implementado corretamente (blank no índice 36, `mask_index` coerente); o fraco desempenho dele é de arquitetura, não de bug.

## 2.2 Modelos e esforço por tarefa (para gastar pouco sem perder qualidade)

| Tarefa | Modelo | Esforço | Motivo |
|---|---|---|---|
| Coordenar, decidir arquitetura e retificação, interpretar resultados e decidir o próximo passo | **Opus 5.5** | alto | decisões difíceis e caras de errar |
| Plano B se o passo 2 não passar de ~60% de placa inteira (redesenho do OCR, estratégia de dados) | **Fable 5.1** (opcional; o usuário autorizou em 12/11, antes estava vetado) | alto | só para o problema mais difícil |
| Passo 0 e 0b (checkpoint, cache, validação completa, R1-R4, curva de confiança) | Sonnet 5.5 | médio | mudanças concretas e testáveis |
| Passo 1 (retificação: `getPerspectiveTransform` no treino, `minAreaRect` na inferência) | Sonnet 5.5 | alto | geometria com risco de erro silencioso; Opus revisa o resultado |
| Passo 2 (implementar a arquitetura decidida pelo Opus) | Sonnet 5.5 | médio | implementação guiada |
| Passo 3 (augmentação, placas sintéticas; checar licença da fonte) | Sonnet 5.5 | médio | |
| Passo 4 (avaliar modelo pré-treinado; checar licença) | Sonnet 5.5 | médio | Opus valida a conclusão |
| Passo 5 (filtro de confiança, votação entre frames, deduplicador) | Sonnet 5.5 | médio | lógica de serviço com testes |
| Rodar treinos/avaliações e registrar números | Sonnet 5.5 | baixo | mecânico, mas lê métricas (sem Haiku em lógica de reconhecimento) |
| README, `.env.example`, renomeações | Haiku 5.5 | baixo | tarefa mecânica |

## 3. Critérios de sucesso (fixar antes de treinar)
- **Dataset (teste completo, 4000)**: placa inteira ponta a ponta >= **70%** (excelente: >= 85%); 3ª posição >= 90%.
- **Precisão do que é enviado ao backend**: entre as leituras que passam no filtro de confiança, >= **98% corretas**, cobrindo >= 60% dos veículos (curva acerto x cobertura).
- **Tempo**: OCR <= 50 ms por recorte em CPU.
- **Webcam/vídeo**: medir à parte; o dataset é o teto otimista.

## 4. Plano (ordem por retorno/custo; medir antes de passar ao próximo)

### Passo 0. Infraestrutura (obrigatório, barato)
- `ModelCheckpoint(save_best_only=True)` a cada época (hoje se perderam ~30 min de treino no desligamento).
- Cache dos recortes pré-processados em `.npz` (carregar 24 000 recortes leva minutos a cada treino).
- **GPU**: TensorFlow nativo no Windows não usa GPU desde a 2.11. Opções: WSL2 + CUDA, ou Google Colab/Kaggle (GPU gratuita). **Antes de subir o dataset para a nuvem, conferir a licença e os termos de uso do RodoSol-ALPR** (liberado mediante pedido, para pesquisa). Com GPU, treinos de horas viram minutos e os passos 2 a 4 ficam viáveis.
- No `evaluate`: acrescentar a **curva confiança x acerto** (para o filtro) e a matriz de confusão (a saída por posição já existe).

### Passo 0b. Correções do código de treino (seção 2.1: R1 a R4)
- Medir antes e depois com o mesmo comando de treino e o `evaluate` completo; registrar o ganho real.

### Passo 1. Retificação da placa (maior ganho esperado pelo custo)
- Treino: usar os **4 cantos** da anotação e `cv2.getPerspectiveTransform` para gerar a placa "de frente" (carro com 1 linha, moto com 2). Na moto, separar as linhas pela geometria retificada, não pelo meio da caixa.
- Inferência: o detector só devolve um retângulo. Opções: (a) `cv2.minAreaRect` no componente da máscara do detector (barato, corrige rotação); (b) detector que prevê os 4 cantos (mais trabalho, corrige perspectiva). Começar pela (a).
- Medir: placa inteira com recorte do gabarito retificado vs não retificado, mesma arquitetura.

### Passo 2. Arquitetura do OCR
- Trocar Flatten + Dense por um backbone que **preserve a largura** (sequência de colunas), com cabeças por posição com atenção, ou **CRNN com BiLSTM + CTC** (o CTC só convolucional falhou, provavelmente por falta de contexto entre as fatias; hipótese não testada).
- Entrada maior e/ou colorida (ex.: 64x192); a cor ajuda a separar Mercosul (faixa azul) de BR.
- Moto: tratar como 2 linhas (duas leituras concatenadas) ou um modelo próprio.

### Passo 3. Dados e augmentação contra o viés
- Augmentação: desfoque (gaussiano e de movimento), ruído, perspectiva leve, brilho/contraste, compressão JPEG.
- **Placas sintéticas** renderizadas com a fonte oficial (Mercosul / FE-Schrift; conferir a licença da fonte), com sequências **uniformes** de letras, para quebrar o viés O/Q/M e R/P/D. Misturar com os dados reais.
- Usar 100% do treino (12 000) com várias versões tremidas; validar com as 4000 de validação completas (hoje só 400).

### Passo 4. Modelo pré-treinado (linha de base para comparar)
- Ex.: `fast-plate-ocr` (Keras 3/ONNX). **Verificar a licença** (LICENSE do repositório) e se algum modelo cobre placas brasileiras/Mercosul. Se sim: avaliar sem ajuste e depois com ajuste fino no RodoSol. Pode ser o caminho mais rápido até >70%.

### Passo 5. Uso em produção (vale para qualquer modelo)
- **Filtro de confiança** escolhido pela curva do passo 0: só envia o que tem alta chance de estar certo.
- **Votação entre frames**: agregar as leituras do mesmo veículo antes de enviar. Atenção: o `Deduplicator` atual envia a **primeira** leitura e bloqueia 30 s; uma leitura errada vira outra "placa" e também passa. Redesenhar como: acumular leituras numa janela curta -> votar por posição -> enviar uma vez.
- **Conferência humana** (decisão de produto do usuário): o vigilante confirma a placa sugerida no front.

## 5. Como medir cada mudança (repetível)
1. Mesmo split (`training/split.py`, seed fixa), mesmo `evaluate`, mesmas 4000 de teste.
2. Registrar no contexto: comando, tempo, sha256, métricas por grupo e por posição.
3. Só promover para `models/ocr.keras` o modelo que vencer o atual em placa inteira ponta a ponta.

## 6. Comandos (estado do código em 12/11)
- `.venv\Scripts\python -m training.train_ocr --arch positions --per-folder 3000 --copies 1 --epochs 30 --out models/ocr_pos_v2.keras` (~2,6 min/época em CPU; repetir depois de adicionar o checkpoint)
- `.venv\Scripts\python -m training.evaluate --ocr <modelo> --out models/<saida>.json`
- `.venv\Scripts\python -m experiments.diagnostico_ocr --ocr <modelo>`
