# 🤖 Tech Challenge - Fase 4: Análise de Vídeo com IA

Este projeto é uma solução para o **Tech Challenge - Fase 4** do curso 4IADT. A aplicação realiza análise automatizada de vídeo com:

- Reconhecimento facial
- Análise de expressões emocionais
- Detecção de atividades e posturas
- Detecção de movimento de membros
- Geração de relatório automatizado em texto

---

## 📂 Estrutura do Projeto

```bash
.
├── tc_video.mp4                 # Vídeo de entrada
├── tc_video_output.mp4          # Vídeo processado com anotações
├── imagens/                     # Recortes salvos das pessoas detectadas
├── relatorio_movimentos_*.txt   # Relatório final gerado automaticamente
├── main.py                      # Script principal do projeto
├── README.md                    # Este arquivo
└── requirements.txt             # Dependências do projeto
```

---

## ⚙️ Requisitos

- Python 3.8 ou superior
- CUDA instalado (opcional, para acelerar o DeepFace com GPU)
- Ambiente virtual recomendado: `venv`, `virtualenv` ou `conda`

Instale as dependências com o comando:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

1. Coloque o vídeo com o nome `tc_video.mp4` na raiz do projeto.
2. Execute o script principal:

```bash
python run_techchallange4.py
```

Durante a execução:

- O vídeo será exibido com anotações.
- Um novo vídeo `tc_video_output.mp4` será salvo.
- Um relatório `.txt` será criado com as análises.
- Os recortes serão salvos na pasta `imagens/`.

---

## 📺 Saída do Projeto

[Google Drive](https://drive.google.com/file/d/1JDq2ZwYoOLiY6hA07gZlj9eR357X8RkP/view?usp=sharing)
- 🎥 Vídeo gerado com anotações: tc_video_output.mp4 
- 📄 Exemplo de relatório gerado: `relatorio_movimentos_20250728_134501.txt`

---

## 📊 Funcionalidades

| Funcionalidade               | Descrição                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| 📦Detecção de Pessoas      | Utiliza YOLOv8                                                |
| 🧍Classificação de Postura | Em pé, sentado ou deitado com base nos pontos da pose        |
| 👋Detecção de Atividades   | Mão levantada, aceno, aperto de mão, dança                 |
| 🧠Análise Emocional         | Identifica emoções com DeepFace (feliz, triste, bravo etc.) |
| 🦿Movimento Corporal         | Verifica movimento de cabeça, braços, pernas, tronco        |
| 📄Relatório Automático     | Gera `.txt` com contagens e percentuais                     |

---

## 🧠Tecnologias Utilizadas

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [MediaPipe - Google](https://google.github.io/mediapipe/)
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- Python 3.8+

---

## 📄Relatório de Saída

O relatório em `.txt` contém:

- Total de quadros processados
- Emoções detectadas por frequência
- Posturas identificadas (em pé, sentado, deitado)
- Atividades detectadas
- Membros com movimento (braços, pernas, etc.)

Exemplo de nome: 

```
relatorio_movimentos_20250728_134501.txt
```

---

## 👥 Equipe

Grupo 7 da turma IA para Devs - FIAP:

* **Fábio Yuiti Takaki** (Discord: `takakisan.`)
* **Luiz Claudio Cunha de Albuquerque** (Discord: `inefavel1305`)
* **Matheus Felipe Condé Rocha** (Discord: `mfconde`)
* **Pedro Vitor Franco de Carvalho** (Discord: `pedro_black10`)
* **Tatiana Yuka Takaki** (Discord: `tatianayk`)
