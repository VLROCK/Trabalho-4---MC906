# Classificação de doenças em folhas de mandioca com robustez a ruído de rótulo

**Projeto MC906 — Trabalho 4**
Adaptação arquitetural do artigo *Large-Scale Fully-Unsupervised Re-Identification*
(Bertocco, Andaló, Rocha; IEEE TBIOM 2025) para um problema de **classificação
supervisionada com rótulos ruidosos**.

---

## 1. Introdução e motivação

O dataset *Cassava Leaf Disease Classification* (Kaggle) tem ~21.397 imagens de
folhas de mandioca em 5 classes (quatro doenças — CBB, CBSD, CGM, CMD — e folha
saudável). É um dataset notoriamente **ruidoso**: os rótulos foram atribuídos por
agricultores usando um aplicativo de campo, com verificação apenas parcial de
especialistas. Ou seja, uma fração desconhecida das imagens está rotulada de
forma inconsistente.

O artigo de referência ataca **Re-Identificação (ReID)** de pessoas e veículos de
forma *totalmente não supervisionada*: não existem rótulos, então o pipeline
gera pseudo-rótulos por clustering e treina os backbones para serem robustos ao
ruído desses pseudo-rótulos. A ligação com o nosso problema é justamente a
**robustez a ruído**: no ReID o ruído vem dos pseudo-rótulos do clustering; no
Cassava ele vem dos rótulos humanos de campo. As técnicas que o artigo usa para
"não deixar o ruído dominar o treino" são a inspiração para este trabalho.

Este projeto tem dois objetivos combinados:

1. **Classificação supervisionada** usando duas ideias arquiteturais do artigo:
   pré-treino self-supervised do backbone (Barlow Twins) e ensemble de
   arquiteturas diversas.
2. **Detecção explícita de erro de rótulo** — algo que o artigo *não* faz (ele
   apenas mitiga o impacto do ruído). Como no Cassava existem rótulos para
   comparar, podemos de fato tentar apontar *quais* imagens têm rótulo suspeito.

---

## 2. Metodologia

### 2.1 O que foi adaptado do artigo (e por quê)

**Pré-treino self-supervised com Barlow Twins.** O artigo pré-treina seus
backbones antes de entrar no pipeline de clustering, para partir de embeddings de
melhor qualidade. Aqui aplicamos a mesma ideia: rodamos **Barlow Twins**
(`ssl_pretrain.py`) sobre as próprias imagens do Cassava, *sem usar os rótulos*.
Isso serve a dois propósitos: (a) dá um bom ponto de partida para o fine-tuning
supervisionado do backbone primário; e (b) produz um **espaço de embeddings** que
usamos depois na detecção de ruído por vizinhança (kNN). O Barlow Twins força a
matriz de cross-correlação entre duas views aumentadas da mesma imagem a se
aproximar da identidade (invariância na diagonal, redução de redundância fora
dela).

**Ensemble de backbones diversos.** Na Seção V-E o artigo defende a
"diversidade complementar": combinar arquiteturas diferentes (ResNet50,
DenseNet121, OSNet) porque elas erram de formas diferentes, e a combinação é mais
robusta que qualquer membro isolado. Adaptamos isso treinando **ResNet50,
DenseNet121 e EfficientNet-B0** separadamente (`train.py`, um run por backbone) e
combinando por **média de probabilidades** na avaliação (`evaluate.py`).
Trocamos o OSNet do artigo por EfficientNet-B0 porque o OSNet é uma rede
projetada especificamente para ReID (features de identidade em múltiplas
escalas), enquanto EfficientNet é uma escolha mais natural e forte para
classificação de imagem genérica. O `evaluate.py` reporta as métricas de **cada
backbone isolado E do ensemble**, justamente para verificar empiricamente se a
diversidade complementar se traduz em ganho — sem os números individuais não é
possível sustentar esse argumento.

### 2.2 O que é exclusivo de ReID e **não** migrou (e por quê)

Vários componentes centrais do artigo só fazem sentido no cenário
*fully-unsupervised* e por isso foram deliberadamente deixados de fora:

- **Proxy/triplet loss por identidade.** No ReID a tarefa é métrica (aproximar
  imagens da mesma identidade, afastar de identidades diferentes). Na
  classificação temos 5 classes fixas e usamos entropia cruzada; não há
  "identidades" para ancorar uma proxy loss.
- **Co-training por permutação de pseudo-rótulos.** O artigo treina múltiplos
  modelos que trocam pseudo-rótulos entre si para reduzir a confirmação de erros
  do clustering. Como aqui os rótulos são dados (ruidosos, mas fixos), não há um
  processo de geração de pseudo-rótulos para permutar.
- **DBSCAN + density scheduling.** Todo o pipeline de clustering (agrupar
  embeddings em identidades com DBSCAN e agendar a densidade ao longo das épocas
  para controlar o ruído dos pseudo-rótulos) existe *porque não há rótulos*. No
  Cassava, os rótulos substituem o clustering: fazemos supervisão direta.

Em uma frase: **do artigo migram os componentes que melhoram a representação
(SSL, ensemble); ficam de fora os componentes que existem para *substituir*
rótulos ausentes (clustering, pseudo-rótulos, proxy loss).**

### 2.3 O que foi criado especificamente para detecção de erro de rótulo

O artigo nunca aponta erros específicos — ele só evita que o ruído domine o
treino. Como temos rótulos para comparar, implementamos três detectores
independentes (`core/`), cada um produzindo um ranking de imagens suspeitas:

1. **Confident Learning** (`cleanlab`, `core/confident_learning.py`). Compara os
   rótulos originais com as probabilidades preditas e ranqueia as amostras cuja
   confiança no próprio rótulo é baixa. É a abordagem mais direta e baseada em
   teoria (estima a matriz de confusão ruído→verdade).
2. **kNN-consistency no espaço SSL** (`core/knn_consistency.py`). Para cada
   imagem, olha seus `k` vizinhos mais próximos no espaço de embeddings do Barlow
   Twins. Se os vizinhos majoritariamente pertencem a outra classe, o rótulo é
   suspeito. É aqui que o pré-treino self-supervised "reaparece" na parte de
   detecção: usamos a geometria aprendida sem rótulos como um juiz independente.
3. **Loss-ranking** (`core/loss_ranking.py`). Amostras cuja loss de treino
   permanece alta ao longo das épocas são candidatas a rótulo errado — o modelo
   aprende as amostras limpas e continua "errando" as inconsistentes. Usamos a
   loss média das últimas épocas (registrada por `train.py` em
   `per_sample_loss.csv`).

Por fim, `core/agreement.py` mede a **concordância** entre os três: Jaccard par a
par e uma tabela de votos (quantos métodos apontam cada imagem). A hipótese é que
amostras sinalizadas por vários métodos independentes são as candidatas mais
fortes a rótulo realmente errado.

### 2.4 Detalhes de implementação

Todo o treino é feito por **scripts executáveis via linha de comando**,
compatíveis com um agendador tipo SLURM (ver `run_all.sh` como referência de
pipeline). `train.py` salva checkpoints periódicos (a cada `N` épocas,
configurável) além do melhor modelo por macro-F1, permitindo retomar jobs
longos. Configurações ficam em `configs/default.yaml` e podem ser sobrescritas
por argumentos. Os notebooks servem **apenas** para visualização pós-treino.

Hardware alvo: cluster com GPU A100 80GB. O código roda em GPU quando disponível
e cai para CPU automaticamente.

---

## 3. Resultados de classificação

> **Observação:** os números abaixo devem ser preenchidos após rodar o pipeline
> no dataset real (`runs/eval/summary.csv`). A tabela e a matriz de confusão são
> geradas automaticamente por `evaluate.py`; o notebook
> `notebooks/visualize_results.ipynb` plota tudo. O pipeline foi validado
> ponta a ponta em um dataset sintético (smoke-test), confirmando que cada etapa
> executa e encadeia corretamente.

Tabela esperada (preencher com `runs/eval/summary.csv`):

| Modelo           | Acurácia | Balanced Acc. | Macro-F1 | Weighted-F1 |
|------------------|----------|---------------|----------|-------------|
| ResNet50 (SSL)   |   —      |      —        |    —     |     —       |
| DenseNet121      |   —      |      —        |    —     |     —       |
| EfficientNet-B0  |   —      |      —        |    —     |     —       |
| **Ensemble**     |   —      |      —        |    —     |     —       |

Análise a incluir: (i) o ensemble supera o melhor membro isolado? Em quanto?
(ii) quais classes concentram os erros na matriz de confusão? No Cassava é comum
a classe majoritária (CMD) dominar e a classe minoritária (CBB) ser a mais
difícil — vale cruzar isso com o F1 por classe do `ensemble_report.json`.

---

## 4. Resultados da análise de label noise

> **Observação:** preencher com `runs/noise/` após rodar `detect_noise.py`.

Itens a reportar:

- **Quantas amostras suspeitas** cada método apontou (top-K configurável em
  `configs/default.yaml`, campo `noise.top_k`).
- **Exemplos visuais** das imagens mais suspeitas de cada método (célula 4 do
  notebook), com o rótulo original ao lado — para inspeção qualitativa.
- **Taxa de concordância** entre os três métodos: Jaccard par a par
  (`agreement.json`) e distribuição de votos (`votes.csv`). Destacar as imagens
  apontadas pelos **três** métodos como as candidatas mais confiáveis.

Interpretação esperada: espera-se concordância **moderada** — os métodos captam
sinais parcialmente diferentes (confiança do modelo, geometria de embeddings,
dificuldade de treino), então uma sobreposição total seria suspeita e uma
sobreposição nula indicaria que estão medindo coisas não relacionadas. A
interseção dos três é o conjunto de maior precisão (menos falsos positivos), ao
custo de menor recall.

---

## 5. Discussão crítica

**Fully-unsupervised (artigo) vs. supervised-com-ruído (Cassava).** A diferença
fundamental é a *fonte* do ruído e o que se pode fazer com ele. No artigo, o
ruído é endógeno ao método (vem do clustering) e não há verdade para comparar, só
resta mitigá-lo (density scheduling, co-training). No Cassava o ruído é exógeno
(vem do rótulo humano) e existe um rótulo para confrontar, o que habilita a
*detecção* explícita — o objetivo 2 deste trabalho, que não tem análogo no
artigo.

**Limitações.**

- **Confident Learning idealmente usa probabilidades out-of-fold.** Na
  implementação atual, para manter o pipeline simples, o CL usa as probabilidades
  do split de validação (onde o modelo não treinou) e o loss-ranking usa o
  conjunto de treino. Isso significa que os três detectores não cobrem exatamente
  o mesmo universo de imagens, e a concordância par a par entre o loss-ranking e
  os demais fica subestimada. A extensão correta é rodar validação cruzada
  (k-fold) para obter probabilidades out-of-fold de *todas* as imagens e então
  aplicar os três métodos sobre o mesmo conjunto — isso torna a análise de
  concordância plenamente comparável. Ficou documentado no código
  (`core/confident_learning.py`) como trabalho recomendado.
- **SSL aplicado a um backbone.** O pré-treino Barlow Twins é feito para o
  backbone primário (que também vira o espaço de embeddings do kNN); os demais
  membros do ensemble entram com init ImageNet. Pré-treinar SSL para cada
  arquitetura seria mais fiel ao artigo, ao custo de ~3× o tempo de pré-treino.
- **Detecção não é correção.** Apontar uma amostra como suspeita não prova que o
  rótulo está errado; a inspeção humana das top-K continua necessária antes de
  remover ou recorrigir qualquer amostra.
- **Ruído dependente de classe.** O ruído do Cassava provavelmente não é
  uniforme (algumas doenças são visualmente mais parecidas e mais confundidas). Os
  métodos baseados em confiança podem herdar o viés do próprio modelo nas classes
  difíceis.

**Ameaças à validade.** O ensemble e o SSL melhoram a representação, mas se o
ruído for sistemático (várias imagens da mesma doença sempre rotuladas como outra
classe específica), tanto o classificador quanto os detectores podem "aprender o
erro" e deixar de sinalizá-lo. Nenhum dos métodos aqui é imune a ruído
correlacionado.

---

## 6. Conclusão

Adaptamos com sucesso dois conceitos arquiteturais de um pipeline de
Re-Identificação totalmente não supervisionado — pré-treino self-supervised
(Barlow Twins) e ensemble de backbones diversos — para um problema diferente:
classificação supervisionada de doenças foliares com rótulos ruidosos.
Deixamos explícito quais componentes do artigo migram (os que melhoram a
representação) e quais são exclusivos do ReID (clustering, pseudo-rótulos, proxy
loss), porque existem apenas para suprir a ausência de rótulos.

Além da classificação, e indo além do que o artigo faz, implementamos a
**detecção explícita de rótulos suspeitos** por três métodos independentes
(Confident Learning, kNN-consistency no espaço SSL e loss-ranking) e uma análise
de concordância entre eles. O pipeline é modular, roda por linha de comando com
checkpointing, e foi validado ponta a ponta. Os resultados quantitativos devem
ser preenchidos após a execução no cluster; a infraestrutura para gerá-los e
visualizá-los está pronta.

---

## Referências

- Bertocco, Andaló, Rocha. *Large-Scale Fully-Unsupervised Re-Identification.*
  IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM), 2025.
- Zbontar, Jing, Misra, LeCun, Deny. *Barlow Twins: Self-Supervised Learning via
  Redundancy Reduction.* ICML 2021.
- Northcutt, Jiang, Chuang. *Confident Learning: Estimating Uncertainty in
  Dataset Labels.* JAIR 2021. (biblioteca `cleanlab`)
- Mwebaze et al. *iCassava 2019 Fine-Grained Visual Categorization Challenge.*
  (dataset Cassava Leaf Disease.)
