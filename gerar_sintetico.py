import json
import os

import cv2
import numpy as np

PASTA = "dados"
LARG_DOC, ALT_DOC = 850, 1100
LARG_FOTO, ALT_FOTO = 1600, 1200

# cantos do documento na foto: sup-esq, sup-dir, inf-dir, inf-esq
CASOS = {
    "frontal_leve": [[420, 110], [1170, 140], [1210, 1110], [380, 1080]],
    "inclinada_lateral": [[300, 200], [1250, 80], [1330, 1150], [260, 1000]],
    "vista_de_baixo": [[560, 150], [1060, 160], [1380, 1120], [230, 1100]],
    "rotacionada": [[620, 60], [1340, 330], [1000, 1150], [250, 860]],
    "forte_perspectiva": [[650, 120], [1000, 110], [1500, 1150], [120, 1160]],
}


def criar_documento():
    doc = np.full((ALT_DOC, LARG_DOC, 3), 250, np.uint8)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(doc, "UFPR - Visao Computacional", (60, 110), fonte, 1.3, (20, 20, 20), 3)
    cv2.putText(doc, "Transformacao perspectiva", (60, 165), fonte, 1.0, (60, 60, 60), 2)
    cv2.line(doc, (60, 195), (LARG_DOC - 60, 195), (0, 0, 0), 2)

    rng = np.random.default_rng(0)
    palavras = ["homografia", "pixel", "matriz", "plano", "ponto", "linha",
                "imagem", "camera", "warp", "opencv", "canto", "folha"]
    for y in range(250, 720, 38):
        x = 60
        while True:
            p = rng.choice(palavras)
            w = cv2.getTextSize(p, fonte, 0.7, 2)[0][0]
            if x + w > LARG_DOC - 60:
                break
            cv2.putText(doc, p, (x, y), fonte, 0.7, (30, 30, 30), 2)
            x += w + 18

    x0, y0, x1, y1 = 60, 760, LARG_DOC - 60, 1040
    for i in range(5):
        cv2.line(doc, (x0, y0 + i * (y1 - y0) // 4), (x1, y0 + i * (y1 - y0) // 4), (0, 0, 0), 2)
        cv2.line(doc, (x0 + i * (x1 - x0) // 4, y0), (x0 + i * (x1 - x0) // 4, y1), (0, 0, 0), 2)
    return doc


def criar_fundo(rng):
    xs = np.arange(LARG_FOTO, dtype=np.float32)
    veios = sum(a * np.sin(xs * f + rng.uniform(0, 6)) for f, a in [(0.01, 18), (0.043, 9), (0.11, 4)])
    ruido = cv2.GaussianBlur(rng.normal(0, 25, (ALT_FOTO, LARG_FOTO)).astype(np.float32), (0, 0), 3)
    base = veios[None, :] + ruido
    return np.clip(np.dstack([70 + base, 105 + base, 145 + base]), 0, 255).astype(np.uint8)


def fotografar(doc, cantos, rng):
    origem = np.float32([[0, 0], [LARG_DOC - 1, 0], [LARG_DOC - 1, ALT_DOC - 1], [0, ALT_DOC - 1]])
    H = cv2.getPerspectiveTransform(origem, np.float32(cantos))
    foto = criar_fundo(rng)
    papel = cv2.warpPerspective(doc, H, (LARG_FOTO, ALT_FOTO), flags=cv2.INTER_CUBIC)
    mascara = cv2.warpPerspective(np.full(doc.shape[:2], 255, np.uint8), H, (LARG_FOTO, ALT_FOTO))
    foto[mascara > 0] = papel[mascara > 0]

    # iluminação mais forte à direita e em cima
    yy, xx = np.mgrid[0:ALT_FOTO, 0:LARG_FOTO].astype(np.float32)
    luz = 0.75 + 0.35 * (xx / LARG_FOTO) * (1 - 0.4 * yy / ALT_FOTO)
    foto = foto * luz[..., None] + rng.normal(0, 6, foto.shape)
    return cv2.GaussianBlur(np.clip(foto, 0, 255).astype(np.uint8), (3, 3), 0)


def main():
    os.makedirs(PASTA, exist_ok=True)
    doc = criar_documento()
    cv2.imwrite(os.path.join(PASTA, "documento_original.png"), doc)
    rng = np.random.default_rng(42)
    for nome, cantos in CASOS.items():
        cv2.imwrite(os.path.join(PASTA, nome + ".jpg"), fotografar(doc, cantos, rng),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(PASTA, "sintetico.json"), "w") as f:
        json.dump({"tamanho_doc": [LARG_DOC, ALT_DOC], "cantos": CASOS}, f, indent=2)


if __name__ == "__main__":
    main()
