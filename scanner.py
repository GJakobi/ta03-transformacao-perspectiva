import argparse
import os

import cv2
import numpy as np

LARGURA_DETECCAO = 800


def ordenar_cantos(pts):
    pts = np.float32(pts).reshape(4, 2)
    # sup-esq tem menor x+y, inf-dir maior x+y; sup-dir tem menor y-x, inf-esq maior y-x
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()
    return pts[[np.argmin(soma), np.argmin(dif), np.argmax(soma), np.argmax(dif)]]


def detectar_cantos(img, etapas=None, refinar=True):
    escala = LARGURA_DETECCAO / img.shape[1]
    peq = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    cinza = cv2.GaussianBlur(cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # fechamento apaga o texto escuro e deixa a folha como uma região clara uniforme
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cinza = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, kernel, iterations=2)

    v = np.median(cinza)
    bordas = cv2.Canny(cinza, int(0.66 * v), int(min(255, 1.33 * v)))
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    quad = None
    for c in contornos:
        if cv2.contourArea(c) < 0.1 * cinza.size:
            break
        aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(aprox) == 4 and cv2.isContourConvex(aprox):
            quad = aprox
            break

    if etapas is not None:
        vis = peq.copy()
        if quad is not None:
            cv2.drawContours(vis, [quad], -1, (0, 255, 0), 3)
        etapas.update(cinza=cinza, bordas=bordas, contorno=vis)

    if quad is None:
        return None
    cantos = ordenar_cantos(quad / escala)
    return refinar_cantos(img, cantos) if refinar else cantos


def refinar_cantos(img, cantos, faixa=12):
    cinza = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cinza = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero(cv2.Canny(cinza, 30, 90))
    pts = np.float32(np.column_stack([xs, ys]))

    retas = []
    for i in range(4):
        p, q = cantos[i], cantos[(i + 1) % 4]
        comp = np.linalg.norm(q - p)
        d = (q - p) / comp
        n = np.array([-d[1], d[0]])
        # pixels de borda perto do lado p-q, ignorando 10% em cada ponta (região dos cantos)
        t = (pts - p) @ d / comp
        perto = pts[(np.abs((pts - p) @ n) < faixa) & (t > 0.1) & (t < 0.9)]
        if len(perto) < 20:
            return cantos
        vx, vy, x0, y0 = cv2.fitLine(perto, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        # reta homogênea que passa por dois pontos = produto vetorial deles
        retas.append(np.cross([x0, y0, 1], [x0 + vx, y0 + vy, 1]))

    # canto i = interseção do lado i-1 com o lado i (produto vetorial das retas)
    novos = [np.cross(retas[i - 1], retas[i]) for i in range(4)]
    novos = np.float32([x[:2] / x[2] for x in novos])
    if np.abs(novos - cantos).max() > 0.1 * max(img.shape[:2]):
        return cantos
    return novos


def selecionar_cantos_manual(img):
    escala = min(1.0, 1000 / max(img.shape[:2]))
    base = cv2.resize(img, None, fx=escala, fy=escala)
    pontos = []

    def clique(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
            pontos.append((x / escala, y / escala))

    janela = "Clique nos 4 cantos (r: refazer, Enter: confirmar, Esc: sair)"
    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, clique)
    while True:
        vis = base.copy()
        for x, y in pontos:
            cv2.circle(vis, (int(x * escala), int(y * escala)), 6, (0, 0, 255), -1)
        if len(pontos) == 4:
            cv2.polylines(vis, [np.int32(ordenar_cantos(pontos) * escala)], True, (0, 255, 0), 2)
        cv2.imshow(janela, vis)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("r"):
            pontos.clear()
        elif tecla in (10, 13) and len(pontos) == 4:
            break
        elif tecla == 27:
            raise SystemExit("Cancelado.")
    cv2.destroyAllWindows()
    return ordenar_cantos(pontos)


def homografia_dlt(origem, destino):
    # cada par (x,y)->(u,v) dá 2 equações lineares em a..h (com h33 = 1): sistema 8x8
    A, b = [], []
    for (x, y), (u, v) in zip(origem, destino):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    return np.append(np.linalg.solve(A, b), 1).reshape(3, 3)


def retificar(img, cantos, largura=None, altura=None):
    if largura is None:
        se, sd, id_, ie = cantos
        largura = int(max(np.linalg.norm(sd - se), np.linalg.norm(id_ - ie)))
        altura = int(max(np.linalg.norm(ie - se), np.linalg.norm(id_ - sd)))
    destino = np.float32([[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]])
    H = cv2.getPerspectiveTransform(np.float32(cantos), destino)
    return cv2.warpPerspective(img, H, (largura, altura), flags=cv2.INTER_CUBIC), H


def efeito_scanner(img):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dilatar + mediana remove o texto e estima o fundo; dividir por ele tira sombras
    fundo = cv2.medianBlur(cv2.dilate(cinza, np.ones((7, 7), np.uint8)), 21)
    plano = cv2.divide(cinza, fundo, scale=255)
    return cv2.adaptiveThreshold(plano, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)


def main():
    ap = argparse.ArgumentParser(description="Scanner de documentos por homografia.")
    ap.add_argument("entrada")
    ap.add_argument("-o", "--saida")
    ap.add_argument("--manual", action="store_true", help="clicar nos 4 cantos")
    ap.add_argument("--cor", action="store_true", help="não binarizar a saída")
    args = ap.parse_args()

    img = cv2.imread(args.entrada)
    if img is None:
        raise SystemExit(f"Não foi possível ler {args.entrada}")

    cantos = selecionar_cantos_manual(img) if args.manual else detectar_cantos(img)
    if cantos is None:
        raise SystemExit("Documento não encontrado. Use --manual.")

    saida, H = retificar(img, cantos)
    if not args.cor:
        saida = efeito_scanner(saida)

    destino = args.saida or os.path.splitext(args.entrada)[0] + "_scan.png"
    cv2.imwrite(destino, saida)
    np.set_printoptions(precision=4, suppress=True)
    print("H =\n", H)
    print("Salvo em", destino)


if __name__ == "__main__":
    main()
