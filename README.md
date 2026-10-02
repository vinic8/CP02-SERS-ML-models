# APIs de Energia Renovável e Aprendizado de Máquina
## Integrantes

* **Vinicius Molena** — RM: 571270
* **Matheus Ferreira** — RM: 569638
* **Nathan Werner** — RM: 572925
* **Gabriel Vilas** — RM: 571603
* **Gustavo Henrique** — RM: 569921
* **Ricardo Santos** — RM: 569600


## Sobre o projeto

Este projeto foi desenvolvido para analisar dados relacionados à geração de energia renovável e à radiação solar utilizando dados obtidos por APIs públicas e técnicas de aprendizado de máquina.

O trabalho está dividido em duas tarefas principais:

1. **Classificação:** classificar a fonte de geração de um empreendimento da ANEEL em **Solar, Eólica ou Hidráulica**, utilizando apenas potência e localização.
2. **Regressão:** estimar a **radiação solar horizontal média**, em W/m², para Petrolina (PE), utilizando variáveis meteorológicas e a hora local.

Em cada tarefa foram treinados e comparados **três algoritmos diferentes**, totalizando seis modelos de aprendizado de máquina.

---

## Estrutura do projeto

```text
.
├── Aula_APIs_Energia_Renovavel_ML_completo.ipynb
├── aneel_classificacao_orange.csv
├── meteo_regressao_orange.csv
└── README.md
```

O notebook contém a consulta às APIs, preparação dos dados, análise exploratória, treinamento dos modelos, métricas, gráficos e conclusões.

---

## Fontes dos dados

### ANEEL — SIGA

Os dados de empreendimentos de geração foram obtidos por meio do **Sistema de Informações de Geração da ANEEL (SIGA)**.

Fonte:

https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel

Recurso utilizado:

https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel/resource/11ec447d-698d-4ab8-977f-b424d5deee6a

Foram consideradas as categorias:

* `UFV` → Solar
* `EOL` → Eólica
* `UHE` → Hidráulica
* `PCH` → Hidráulica
* `CGH` → Hidráulica

### Open-Meteo

Os dados meteorológicos históricos foram obtidos pela API do **Open-Meteo**.

Fonte:

https://open-meteo.com/en/docs/historical-weather-api

Local utilizado:

* Latitude: `-9.39`
* Longitude: `-40.50`
* Localidade aproximada: Petrolina (PE)
* Fuso horário: `America/Recife`

Período consultado:

**01/04/2025 a 30/06/2025**

Foram utilizados dados horários e consideradas as horas entre **07h e 17h**.

---

# Tarefa 1 — Classificação

## Objetivo

Verificar se é possível classificar a fonte de geração de um empreendimento como **Solar, Eólica ou Hidráulica** utilizando somente:

* potência outorgada;
* latitude;
* longitude.

O objetivo não é prever a quantidade de energia produzida, mas sim classificar a categoria da fonte.

## Entradas e alvo

As entradas `X` possuem três variáveis:

| Variável      | Descrição                                   |
| ------------- | ------------------------------------------- |
| `potencia_kw` | Potência outorgada do empreendimento, em kW |
| `latitude`    | Latitude aproximada do empreendimento       |
| `longitude`   | Longitude aproximada do empreendimento      |

O alvo `y` é:

```text
fonte
```

com três classes:

```text
Solar
Eólica
Hidráulica
```

Não foram utilizadas como entradas variáveis como `SigTipoGeracao`, nome do empreendimento, CEG ou descrições da fonte.

## Tratamento e divisão dos dados

Os dados foram divididos em:

* **80% para treinamento**
* **20% para teste**

A divisão foi realizada de forma **estratificada**, mantendo a proporção das classes nos conjuntos de treino e teste.

Foi utilizada a semente:

```text
random_state = 42
```

Para os modelos que utilizam escala, o `StandardScaler` foi colocado dentro de um `Pipeline`, garantindo que a padronização fosse aprendida somente com os dados de treinamento.

## Classificadores utilizados

### K-Nearest Neighbors — KNN

O KNN classifica um novo empreendimento observando os exemplos mais próximos no espaço das características. É adequado para comparar como a proximidade entre potência e localização pode separar as classes.

### Regressão Logística

A Regressão Logística foi utilizada como um modelo de referência com relações mais simples entre as variáveis de entrada e as classes.

### Random Forest

O Random Forest combina diversas árvores de decisão e consegue representar relações não lineares entre potência, latitude e longitude.

## Métricas

Os modelos foram comparados utilizando:

* **Accuracy**
* **Precision**
* **Recall**
* **F1**

Para Precision, Recall e F1 foi utilizada a **média macro (`macro`)**, ou seja, cada uma das três classes recebe o mesmo peso no cálculo final.

Também foi apresentada a **matriz de confusão** de cada modelo.

As tabelas e matrizes de confusão podem ser encontradas diretamente no notebook.

## Interpretação

A comparação dos modelos permite observar como diferentes algoritmos se comportam quando recebem somente potência e localização.

A análise também mostra quais classes apresentam maior dificuldade de separação. A matriz de confusão permite identificar os casos em que, por exemplo, um empreendimento de determinada fonte foi classificado como outra.

Potência e localização são úteis como características iniciais, mas podem não ser suficientes para uma aplicação real. Empreendimentos de fontes diferentes podem possuir potências semelhantes e estar localizados em regiões próximas.

Uma classificação mais completa poderia considerar outras informações técnicas e cadastrais do empreendimento, desde que essas informações estejam disponíveis no momento da previsão e sejam válidas para o uso pretendido.

---

# Tarefa 2 — Regressão

## Objetivo

Estimar a **radiação solar horizontal média** em Petrolina (PE), utilizando variáveis meteorológicas e a hora local.

O alvo da regressão é:

```text
radiacao_w_m2
```

medido em:

```text
W/m²
```

## Variáveis utilizadas

As cinco entradas `X` são:

| Variável        | Descrição                           |
| --------------- | ----------------------------------- |
| `temperatura_c` | Temperatura do ar a 2 m, em °C      |
| `umidade_pct`   | Umidade relativa do ar, em %        |
| `nuvens_pct`    | Cobertura de nuvens, em %           |
| `vento_kmh`     | Velocidade do vento a 10 m, em km/h |
| `hora`          | Hora local do registro              |

O alvo `y` é:

```text
radiacao_w_m2
```

A variável `data_hora` é utilizada para manter os registros em ordem cronológica e realizar a separação entre treino e teste, mas não é utilizada diretamente como entrada do modelo.

Nenhum valor de `radiacao_w_m2` do próprio registro foi utilizado para criar uma entrada do mesmo registro.

## Tratamento e divisão temporal

Os registros foram mantidos em ordem crescente de `data_hora`.

A divisão utilizada foi aproximadamente:

* **80% das primeiras horas → treinamento**
* **20% das horas finais → teste**

Não foi realizado embaralhamento dos registros.

Para a Regressão Linear foi utilizado `StandardScaler` dentro de um `Pipeline`, garantindo que os parâmetros de escala fossem aprendidos somente a partir do conjunto de treinamento.

Os modelos baseados em árvores não necessitam de padronização.

## Regressores utilizados

### Regressão Linear

Foi utilizada como modelo de referência para verificar a relação aproximadamente linear entre as variáveis meteorológicas, a hora e a radiação.

### Árvore de Decisão

A Árvore de Decisão permite representar relações não lineares e divisões diferentes entre as características meteorológicas.

### Random Forest Regressor

O Random Forest Regressor combina várias árvores de decisão e permite representar relações mais complexas entre as variáveis de entrada e a radiação solar.

## Métricas

Os modelos foram comparados utilizando:

### MAE

Erro absoluto médio, apresentado em:

```text
W/m²
```

Quanto menor o MAE, menor tende a ser o erro médio das previsões.

### MSE

Erro quadrático médio, apresentado em:

```text
(W/m²)²
```

O MSE dá maior peso aos erros grandes.

### R²

Métrica que indica quanto da variabilidade observada no conjunto de teste é explicada pelo modelo na divisão utilizada.

Além da tabela comparativa, o notebook apresenta um gráfico de **valores reais × valores previstos**.

## Interpretação

A variável `hora` pode ter importância relevante porque a radiação solar apresenta um comportamento diário. O horário ajuda o modelo a identificar em qual momento do ciclo solar o registro está localizado.

Mesmo assim, a hora sozinha não é suficiente para explicar todas as variações da radiação. Condições como cobertura de nuvens, umidade, temperatura e vento também podem contribuir para as diferenças observadas.

A comparação entre MAE, MSE e R² permite analisar os modelos por diferentes perspectivas:

* MAE mostra o erro típico em W/m²;
* MSE evidencia mais fortemente erros grandes;
* R² mostra a capacidade de explicar a variabilidade da radiação no conjunto de teste.

---

# Radiação solar não é igual à energia gerada

A variável `radiacao_w_m2` representa radiação por unidade de área e por tempo, não a quantidade final de energia elétrica produzida por um sistema fotovoltaico.

Para estimar a energia gerada seria necessário considerar outras características, como:

* área dos módulos;
* eficiência dos painéis;
* orientação e inclinação;
* temperatura dos módulos;
* perdas do sistema;
* eficiência do inversor;
* características elétricas do sistema;
* duração da exposição à irradiância.

Portanto, uma previsão de radiação em W/m² **não corresponde automaticamente à energia produzida em kWh**.

---

# Resultados

O notebook apresenta as seis comparações exigidas pela atividade.

## Classificação

A tabela de classificação apresenta, para os três classificadores:

| Algoritmo           |     Accuracy | Precision (macro) | Recall (macro) |   F1 (macro) |
| ------------------- | -----------: | ----------------: | -------------: | -----------: |
| KNN                 | Ver notebook |      Ver notebook |   Ver notebook | Ver notebook |
| Regressão Logística | Ver notebook |      Ver notebook |   Ver notebook | Ver notebook |
| Random Forest       | Ver notebook |      Ver notebook |   Ver notebook | Ver notebook |

Também são apresentadas as matrizes de confusão dos três modelos para identificar as classes mais confundidas.

## Regressão

A tabela de regressão apresenta:

| Algoritmo         |   MAE (W/m²) | MSE ((W/m²)²) |           R² |
| ----------------- | -----------: | ------------: | -----------: |
| Regressão Linear  | Ver notebook |  Ver notebook | Ver notebook |
| Árvore de Decisão | Ver notebook |  Ver notebook | Ver notebook |
| Random Forest     | Ver notebook |  Ver notebook | Ver notebook |

O notebook também apresenta o gráfico de **real × previsto** e a importância das variáveis no Random Forest.

Os valores finais das métricas são calculados diretamente a partir dos dados utilizados no notebook, evitando registrar resultados que possam ficar diferentes caso os CSVs sejam reproduzidos novamente.

---

# Como executar o projeto

## 1. Instalar o Python

Recomenda-se utilizar **Python 3.10 ou superior**.

## 2. Instalar as bibliotecas

No terminal:

```bash
pip install pandas numpy matplotlib scikit-learn requests jupyter
```

## 3. Abrir o notebook

Execute:

```bash
jupyter notebook
```

ou:

```bash
jupyter lab
```

Depois, abra:

```text
Aula_APIs_Energia_Renovavel_ML_completo.ipynb
```

## 4. Executar as células

As células devem ser executadas **na ordem**, começando pelo início do notebook.

O notebook consulta:

* a API da ANEEL para os dados de classificação;
* a API histórica do Open-Meteo para os dados de regressão.

As consultas não utilizam senhas ou tokens.

Durante a execução, são gerados os arquivos:

```text
aneel_classificacao_orange.csv
meteo_regressao_orange.csv
```

Caso os CSVs já estejam no repositório, eles também podem ser carregados diretamente pelo notebook.

---

# Arquivos do projeto

### `Aula_APIs_Energia_Renovavel_ML_completo.ipynb`

Notebook principal contendo:

* consultas às APIs;
* preparação dos dados;
* análise exploratória;
* classificação;
* regressão;
* métricas;
* matrizes de confusão;
* gráficos;
* interpretação dos resultados;
* conclusões.

### `aneel_classificacao_orange.csv`

Dataset utilizado na tarefa de classificação.

Colunas:

```text
potencia_kw
latitude
longitude
fonte
```

### `meteo_regressao_orange.csv`

Dataset utilizado na tarefa de regressão.

Colunas:

```text
data_hora
temperatura_c
umidade_pct
nuvens_pct
vento_kmh
hora
radiacao_w_m2
```

---

# Conclusão

O projeto demonstra a aplicação de aprendizado de máquina em dois problemas diferentes relacionados à energia renovável.

Na **classificação**, são comparados KNN, Regressão Logística e Random Forest para verificar como potência e localização conseguem separar as classes Solar, Eólica e Hidráulica.

Na **regressão**, são comparados Regressão Linear, Árvore de Decisão e Random Forest para estimar a radiação solar utilizando variáveis meteorológicas e a hora local.

As métricas, tabelas, matrizes de confusão, gráficos e interpretações apresentados no notebook permitem analisar o comportamento dos modelos e suas limitações.

O projeto também destaca a importância de evitar **vazamento de dados**, respeitar a ordem temporal na regressão e diferenciar uma estimativa de radiação solar da previsão direta de energia elétrica produzida por um sistema fotovoltaico.
