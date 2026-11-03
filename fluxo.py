"""Fluxo Kaggle + GitHub + W&B dos notebooks: acha o CSV, loga os trials e guarda os resultados.

Todos sao opcionais: sem token, os resultados ficam so em disco.
"""
import os, glob, shutil, subprocess, tempfile, time

NOME_CSV = "customer_churn_telecom_services.csv"
DATASET = "kapturovalexander/customers-churned-in-telecom-services"


def achar_csv(caminho=""):
    """Ordem: caminho dado -> CHURN_CSV -> dados/telco_ibm.csv -> Input do Kaggle -> download via kagglehub."""
    if caminho and not os.path.isfile(caminho):
        raise FileNotFoundError(f"CSV_PATH nao encontrado: {caminho}")
    raiz = os.path.dirname(os.path.abspath(__file__))
    for c in (caminho, os.environ.get("CHURN_CSV", ""), os.path.join(raiz, "dados", "telco_ibm.csv")):
        if c and os.path.isfile(c):
            return c
    achados = glob.glob(f"/kaggle/input/**/{NOME_CSV}", recursive=True)
    if achados:
        return achados[0]
    for c in glob.glob("/kaggle/input/**/*.csv", recursive=True):       # dataset proprio com outro nome de arquivo
        with open(c, encoding="utf-8", errors="ignore") as f:
            cabecalho = f.readline()
        if "Churn" in cabecalho and "tenure" in cabecalho:
            return c
    import kagglehub
    return os.path.join(kagglehub.dataset_download(DATASET), NOME_CSV)


def _url(repo_url, token):
    return repo_url.replace("https://", f"https://x-access-token:{token}@", 1) if token and repo_url.startswith("https://") else repo_url


def _git(args, token="", cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    msg = (r.stdout + r.stderr).replace(token, "***") if token else r.stdout + r.stderr   # o token nunca vai para o log
    return r.returncode, msg


def salvar_github(pasta, run, repo_url, token="", branch="resultados"):
    """Copia `pasta` para <run>/ na branch de resultados (orfa, criada se nao existir). Falha nao derruba o treino."""
    if not repo_url:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        url = _url(repo_url, token)
        if _git(["clone", "-q", "--depth", "1", "--branch", branch, url, tmp], token)[0] != 0:
            shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
            _git(["init", "-q"], cwd=tmp); _git(["checkout", "-q", "--orphan", branch], cwd=tmp)
            _git(["remote", "add", "origin", url], token, tmp)
        destino = os.path.join(tmp, run)
        shutil.copytree(pasta, destino, dirs_exist_ok=True)
        _git(["add", "-A"], cwd=tmp)
        if _git(["diff", "--cached", "--quiet"], cwd=tmp)[0] == 0 and _git(["rev-parse", "HEAD"], cwd=tmp)[0] == 0:
            return True
        _git(["-c", "user.name=churn-bot", "-c", "user.email=churn-bot@users.noreply.github.com", "-c", "commit.gpgsign=false", "commit", "-qm", f"resultados: {run}"], cwd=tmp)
        cod, msg = _git(["push", "-q", "origin", f"HEAD:{branch}"], token, tmp)
        if cod != 0:
            print("GitHub: falha ao enviar os resultados:", msg[-300:])
        return cod == 0


def baixar_github(pasta, run, repo_url, token="", branch="resultados"):
    """Traz os resultados de uma execucao anterior, para retomar o estudo."""
    with tempfile.TemporaryDirectory() as tmp:
        cod, msg = _git(["clone", "-q", "--depth", "1", "--branch", branch, _url(repo_url, token), tmp], token)
        origem = os.path.join(tmp, run)
        if cod != 0 or not os.path.isdir(origem):
            print(f"GitHub: nada para retomar em {branch}/{run}.")
            return False
        shutil.copytree(origem, pasta, dirs_exist_ok=True)
        return True


def callback_trial(wandb_run=None, salvar=None, backup_min=30):
    """Callback do Optuna: loga cada trial no W&B e chama `salvar` a cada backup_min minutos."""
    ultimo = [time.time()]

    def cb(st, t):
        if wandb_run is not None:
            bons = [x.value for x in st.trials if x.state.name == "COMPLETE"]
            wandb_run.log({"trial": t.number, "estado": t.state.name, "ks": t.value, "melhor_ks": max(bons) if bons else None,
                           **{f"hp/{k}": v for k, v in t.params.items()}})
        if salvar and time.time() - ultimo[0] > backup_min * 60:
            salvar(); ultimo[0] = time.time()
    return cb
