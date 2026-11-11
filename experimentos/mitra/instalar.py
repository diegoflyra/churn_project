"""Instala o Mitra (autogluon.tabular[mitra]) sem trocar o torch que o Kaggle casou com a GPU.

O autogluon.tabular 1.6.3 pede torch>=2.10,<2.14, pandas<2.4 e numpy<2.6. Antes de instalar, o pip simula a instalacao
(--dry-run --report) e, se fosse trocar o torch, para com uma mensagem em vez de instalar. So usa a biblioteca padrao,
porque roda antes de qualquer outro import do notebook.
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tempfile

REQUISITOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
PROTEGIDOS = ("torch", "torchvision", "torchaudio", "triton")


def _versao(pacote):
    try:
        return md.version(pacote)
    except md.PackageNotFoundError:
        return None


def avaliar_relatorio(relatorio, instalado=_versao):
    """O que o pip instalaria (pacote -> versao) e quais pacotes protegidos ja instalados ele trocaria."""
    novos = {i["metadata"]["name"].lower().replace("_", "-"): i["metadata"]["version"] for i in relatorio.get("install", [])}
    trocas = {p: (instalado(p), novos[p]) for p in PROTEGIDOS if p in novos and instalado(p) is not None}
    if any(n.startswith("nvidia-") for n in novos) and instalado("torch") is not None:
        trocas.setdefault("torch (pacotes nvidia-*)", (instalado("torch"), novos.get("torch", "?")))
    return novos, trocas


def instalar(requisitos=REQUISITOS, permitir_trocar_torch=False):
    """Instala o que falta. Devolve {pacote: versao} do que foi instalado ({} se ja estava tudo)."""
    if _versao("autogluon.tabular") == "1.6.3" and _versao("einx") is not None:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        arq = os.path.join(tmp, "relatorio.json")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--dry-run", "--report", arq, "-r", requisitos])
        with open(arq, encoding="utf-8") as f:
            novos, trocas = avaliar_relatorio(json.load(f))
    if trocas and not permitir_trocar_torch:
        detalhe = ", ".join(f"{p} {a} -> {b}" for p, (a, b) in trocas.items())
        raise RuntimeError(f"Instalar o Mitra trocaria o torch do ambiente ({detalhe}). O autogluon.tabular 1.6.3 pede "
                           "torch>=2.10,<2.14. Use um ambiente com torch nessa faixa ou, sabendo que a sessao precisa ser "
                           "reiniciada e que a GPU pode deixar de funcionar, rode com PERMITIR_TROCAR_TORCH = True.")
    ja_importados = sorted(p for p in novos if p.replace("-", "_") in sys.modules)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", requisitos])
    if ja_importados:
        raise RuntimeError(f"Instalado, mas {ja_importados} ja estavam carregados com a versao antiga. "
                           "Reinicie a sessao (Run > Restart & clear cell outputs) e rode de novo.")
    return novos
