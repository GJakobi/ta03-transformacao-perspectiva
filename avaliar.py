import json
import os

import cv2
import numpy as np

import scanner

DADOS = "dados"
FIGS = os.path.join("relatorio", "figuras")
JPG = [cv2.IMWRITE_JPEG_QUALITY, 85]


def painel(imgs, altura=600):
    linha = []
    for img, titulo in imgs:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (round(img.shape[1] * altura / img.shape[0]), altura),
                         interpolation=cv2.INTER_AREA)
        topo = np.full((46, img.shape[1], 3), 255, np.uint8)
        cv2.putText(topo, titulo, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        linha += [np.vstack([topo, img]), np.full((altura + 46, 12, 3), 255, np.uint8)]
    return np.hstack(linha[:-1])


def ncc(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float(np.mean(a * b))


def ncc_retificada(foto, cantos, original):
    ret, _ = scanner.retificar(foto, cantos, original.shape[1], original.shape[0])
    return ncc(cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY).astype(float), original)


def erro_cantos(cantos, gt):
    return np.linalg.norm(cantos - gt, axis=1).mean()


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(DADOS, "sintetico.json")) as f:
        meta = json.load(f)
    larg, alt = meta["tamanho_doc"]
    original = cv2.imread(os.path.join(DADOS, "documento_original.png"),
                          cv2.IMREAD_GRAYSCALE).astype(float)
    destino = np.float32([[0, 0], [larg - 1, 0], [larg - 1, alt - 1], [0, alt - 1]])

    tabela = []
    for nome, gt in meta["cantos"].items():
        foto = cv2.imread(os.path.join(DADOS, nome + ".jpg"))
        gt = np.float32(gt)
        etapas = {}
        cantos = scanner.detectar_cantos(foto, etapas)
        if cantos is None:
            tabela.append(f"{nome} & \\multicolumn{{6}}{{c}}{{falhou}} \\\\")
            continue
        grosso = scanner.detectar_cantos(foto, refinar=False)

        H_cv = cv2.getPerspectiveTransform(cantos, destino)
        dif_H = np.abs(scanner.homografia_dlt(cantos, destino) - H_cv).max()

        linha = [erro_cantos(grosso, gt), erro_cantos(cantos, gt),
                 ncc_retificada(foto, grosso, original), ncc_retificada(foto, cantos, original),
                 ncc_retificada(foto, gt, original)]
        print(nome, " ".join(f"{v:.3f}" for v in linha), f"{dif_H:.0e}")
        tabela.append(nome.replace("_", r"\_") + " & "
                      + " & ".join(f"{v:.2f}" for v in linha[:2]) + " & "
                      + " & ".join(f"{v:.3f}" for v in linha[2:]) + f" & {dif_H:.0e} \\\\")

        ret, _ = scanner.retificar(foto, cantos)
        vis = foto.copy()
        cv2.polylines(vis, [np.int32(cantos)], True, (0, 255, 0), 6)
        cv2.imwrite(os.path.join(FIGS, f"caso_{nome}.jpg"),
                    painel([(vis, "Foto + cantos"), (ret, "Retificada"),
                            (scanner.efeito_scanner(ret), "Scanner")]), JPG)

        if nome == "inclinada_lateral":
            cv2.imwrite(os.path.join(FIGS, "pipeline.jpg"),
                        painel([(foto, "(a) Entrada"), (etapas["cinza"], "(b) Cinza+fech."),
                                (etapas["bordas"], "(c) Canny"),
                                (etapas["contorno"], "(d) Quadrilatero")], altura=420), JPG)

            A = cv2.getAffineTransform(gt[[0, 1, 3]], destino[[0, 1, 3]])
            afim = cv2.warpAffine(foto, A, (larg, alt), flags=cv2.INTER_CUBIC)
            persp, _ = scanner.retificar(foto, gt, larg, alt)
            cv2.imwrite(os.path.join(FIGS, "afim_vs_perspectiva.jpg"),
                        painel([(afim, "Afim (3 pontos)"),
                                (persp, "Perspectiva (4 pontos)")], altura=550), JPG)

    with open(os.path.join("relatorio", "tabela.tex"), "w") as f:
        f.write("\n".join(tabela) + "\n")

    foto = cv2.imread(os.path.join(DADOS, "foto_real.jpg"))
    if foto is not None:
        etapas = {}
        cantos = scanner.detectar_cantos(foto, etapas)
        ret, _ = scanner.retificar(foto, cantos)
        vis = foto.copy()
        cv2.polylines(vis, [np.int32(cantos)], True, (0, 255, 0), 8)
        cv2.imwrite(os.path.join(FIGS, "foto_real.jpg"),
                    painel([(etapas["bordas"], "(a) Canny"), (vis, "(b) Cantos (Hough)"),
                            (ret, "(c) Retificada"), (scanner.efeito_scanner(ret), "(d) Scanner")],
                           altura=700), JPG)


if __name__ == "__main__":
    main()
