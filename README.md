# Scanner de documentos com transformação perspectiva

Trabalho de Visão Computacional (UFPR, Prof. Eduardo Todt).
Gustavo Jakobi (GRR20221253) e Bruno Crestani (GRR20221240).
O programa recebe a foto de um documento tirada em ângulo e devolve o documento
"escaneado": visto de frente, retangular e em preto e branco.

## Como funciona

1. Detecta a folha: escala de cinza, fechamento morfológico, Canny e `findContours`.
2. Aproxima o maior contorno por um quadrilátero (`approxPolyDP`).
3. Refina os cantos: ajusta uma reta a cada lado (`fitLine`) e intersecta as retas.
4. Calcula a homografia com `cv2.getPerspectiveTransform` (4 pares de pontos).
5. Aplica `cv2.warpPerspective`.
6. Opcional: aplica o efeito de scanner (remoção de sombra e `adaptiveThreshold`).

O arquivo `scanner.py` também resolve o sistema linear 8x8 da homografia
(`homografia_dlt`), para comparar com o resultado do OpenCV.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scanner.py foto.jpg                  # automático
.venv/bin/python scanner.py foto.jpg --manual         # clicar nos 4 cantos
.venv/bin/python scanner.py foto.jpg --cor          # só retifica, sem binarizar
```

## Reproduzir a avaliação

```bash
.venv/bin/python gerar_sintetico.py   # gera fotos sintéticas em dados/
.venv/bin/python avaliar.py           # mede erros e gera relatorio/figuras/
cd relatorio && pdflatex relatorio.tex && pdflatex relatorio.tex
```

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `scanner.py` | Aplicação (detecção, homografia, warp, efeito de scanner) |
| `gerar_sintetico.py` | Fotos de teste com cantos conhecidos |
| `avaliar.py` | Métricas e figuras do relatório |
| `relatorio/` | Relatório em formato de artigo (LaTeX) |
