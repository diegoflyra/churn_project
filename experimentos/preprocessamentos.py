"""Confere os registros de pre-processamento e gera a comparacao entre modelos.

Uso: python experimentos/preprocessamentos.py
Le:  experimentos/<modelo>/preprocessamento/registro.json
Gera: experimentos/PREPROCESSAMENTOS.md
"""
import glob, json, os, sys

PASTA = os.path.dirname(os.path.abspath(__file__))
DECISOES = {"base", "aceita", "rejeitada", "em teste"}
ETAPAS = ["divisao_validacao", "variaveis_criadas", "scaler", "oversampling", "extras"]


def estado(reg, id_versao):
    """Etapas em vigor numa versao: a base com as alteracoes de cada versao ate ela."""
    por_id = {v["id"]: v for v in reg["versoes"]}
    cadeia = []
    atual = por_id[id_versao]
    while atual:
        cadeia.append(atual)
        atual = por_id.get(atual.get("parte_de"))
    e = {}
    for v in reversed(cadeia):
        for k, x in v.get("altera", {}).items():
            e[k] = e.get(k, []) + x if k == "extras" else x
    return e


def conferir(reg, arq):
    erros, ids = [], set()
    for v in reg["versoes"]:
        for campo in ("id", "mudanca", "decisao", "codigo"):
            if not v.get(campo):
                erros.append(f"{arq}: {v.get('id', '?')} sem o campo {campo}")
        if v.get("decisao") not in DECISOES:
            erros.append(f"{arq}: {v.get('id')} com decisao invalida ({v.get('decisao')})")
        if v.get("parte_de") and v["parte_de"] not in ids:
            erros.append(f"{arq}: {v.get('id')} parte de uma versao anterior inexistente ({v['parte_de']})")
        ids.add(v.get("id"))
    base = [v for v in reg["versoes"] if v.get("decisao") == "base"]
    if len(base) != 1:
        erros.append(f"{arq}: precisa de exatamente uma versao base")
    if reg.get("atual") not in ids:
        erros.append(f"{arq}: versao atual inexistente ({reg.get('atual')})")
    return erros


def _ks(x):
    return "-" if x is None else f"{x:.4f}"


def _sinal(x):
    return "-" if x is None else f"{x:+.4f}"


def gerar(regs):
    L = ["# Comparacao dos pre-processamentos", "",
         "Gerado por `python experimentos/preprocessamentos.py`. Nao edite aqui: edite o `registro.json` de cada modelo.",
         "As regras estao em `experimentos/README.md`.", "", "## O que cada modelo usa hoje", "",
         "| Modelo | Versao | Validacao | Variaveis criadas | Scaler | Oversampling | Extras | Assinatura do teste |", "|---|---|---|---|---|---|---|---|"]
    sim = lambda x: "sim" if x is True else "nao" if x is False else "-" if x is None else str(x)
    for m, r in regs:
        e = estado(r, r["atual"])
        L.append(f"| {m} | {r['atual']} | {e.get('divisao_validacao', '-')} | {sim(e.get('variaveis_criadas'))} | {sim(e.get('scaler'))} | "
                 f"{sim(e.get('oversampling'))} | {', '.join(e.get('extras', [])) or '-'} | {r.get('assinatura_teste', '-')} |")
    L += ["", "## Evolucao por modelo", "", "So compare numeros de KS com o mesmo protocolo. O teste nao aparece aqui: ele fica para a avaliacao final.", ""]
    for m, r in regs:
        L += [f"### {m}", "", "| Versao | Parte de | Mudanca | KS | Delta | Ganhou | Regua | Decisao | Protocolo |", "|---|---|---|---|---|---|---|---|---|"]
        for v in r["versoes"]:
            a = v.get("validacao") or {}
            L.append(f"| {v['id']} | {v.get('parte_de') or '-'} | {v['mudanca']} | {_ks(a.get('ks'))} | {_sinal(a.get('delta'))} | {a.get('ganhou', '-')} | "
                     f"{_ks(a.get('regua'))} | {v['decisao']} | {a.get('protocolo', '-')} |")
        L.append("")
    return "\n".join(L)


def main():
    arqs = sorted(glob.glob(os.path.join(PASTA, "*", "preprocessamento", "registro.json")))
    regs, erros = [], []
    for a in arqs:
        r = json.load(open(a, encoding="utf-8"))
        erros += conferir(r, os.path.relpath(a, PASTA))
        regs.append((r["modelo"], r))
    if erros:
        sys.exit("Registros com problema:\n" + "\n".join(erros))
    assinaturas = {r["assinatura_teste"] for _, r in regs if r.get("assinatura_teste")}
    if len(assinaturas) > 1:
        sys.exit(f"Assinaturas de teste diferentes: {sorted(assinaturas)}. O teste precisa ser o mesmo para todos.")
    open(os.path.join(PASTA, "PREPROCESSAMENTOS.md"), "w", encoding="utf-8", newline="\n").write(gerar(regs) + "\n")
    print(f"{len(regs)} registros conferidos; experimentos/PREPROCESSAMENTOS.md atualizado.")


if __name__ == "__main__":
    main()
