"""Treino e avaliação da KAN pykan 0.2.8 para o experimento de churn."""

import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from kan import KAN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import comum


DEVICE = os.environ.get(
    "CHURN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
torch.set_num_threads(1)

KAN_PADRAO = {
    "hidden": (8,),
    "mult_nodes": 0,
    "mult_arity": 2,
    "grid": 3,
    "k": 3,
    "base_fun": "silu",
    "grid_eps": 0.02,
    "opt": "Adam",
    "lr": 1e-3,
    "steps": 100,
    "batch": -1,
    "update_grid": False,
    "grid_update_num": 10,
    "start_grid_update_step": -1,
    "stop_grid_update_step": 50,
    "lamb": 0.0,
    "lamb_l1": 1.0,
    "lamb_entropy": 2.0,
    "lamb_coef": 0.0,
    "lamb_coefdiff": 0.0,
    "scale_base_mu": 0.0,
    "scale_base_sigma": 1.0,
    "sb_trainable": True,
    "sp_trainable": True,
}

KAN_TRIAL0 = {
    "arquitetura": "8",
    "grid": 3,
    "k": 3,
    "base_fun": "silu",
    "opt": "LBFGS",
    "lr_lbfgs": 1.0,
    "steps": 100,
    "batch": -1,
    "grid_eps": 0.02,
}


def _width(hp, n_inputs):
    hidden = list(hp.get("hidden", KAN_PADRAO["hidden"]))
    mult_nodes = int(hp.get("mult_nodes", 0))
    if mult_nodes:
        layers = [[int(nodes), mult_nodes if i == 0 else 0] for i, nodes in enumerate(hidden)]
    else:
        layers = [int(nodes) for nodes in hidden]
    return [int(n_inputs), *layers, 1]


def _nome_otimizador(hp):
    nome = str(hp.get("opt", KAN_PADRAO["opt"])).strip().casefold()
    otimizadores = {"adam": "Adam", "lbfgs": "LBFGS"}
    if nome not in otimizadores:
        raise ValueError(
            f"Otimizador KAN não suportado: {nome!r}. Use 'Adam' ou 'LBFGS'."
        )
    return otimizadores[nome]


def verificar_estado(model, etapa):
    invalidos = []

    for nome, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            invalidos.append(nome)

    print(f"[{etapa}] Tensores inválidos: {invalidos}")

    if invalidos:
        raise FloatingPointError(
            f"{etapa}: tensores não finitos: {', '.join(invalidos)}"
        )


def verificar_historico(history, etapa):
    for nome in ("train_loss", "test_loss"):
        if nome not in history:
            raise RuntimeError(f"{etapa}: histórico PyKAN sem a chave {nome!r}.")
        valores = np.asarray(history[nome], dtype=float)
        if not valores.size:
            raise RuntimeError(f"{etapa}: histórico PyKAN vazio para {nome}.")
        invalidos = np.flatnonzero(~np.isfinite(valores))
        if invalidos.size:
            raise FloatingPointError(
                f"{etapa}: {nome} não finita nos passos "
                f"{invalidos.tolist()}."
            )

    if "reg" in history:
        reg = np.asarray(history["reg"], dtype=float)
        invalidos = np.flatnonzero(~np.isfinite(reg))
        if invalidos.size:
            raise FloatingPointError(
                f"{etapa}: regularização não finita nos passos "
                f"{invalidos.tolist()}."
            )


def criar_modelo(hp, n_inputs, seed, save_act=True):
    return KAN(
        width=_width(hp, n_inputs),
        grid=int(hp.get("grid", KAN_PADRAO["grid"])),
        k=int(hp.get("k", KAN_PADRAO["k"])),
        mult_arity=int(hp.get("mult_arity", KAN_PADRAO["mult_arity"])),
        base_fun=hp.get("base_fun", KAN_PADRAO["base_fun"]),
        grid_eps=float(hp.get("grid_eps", KAN_PADRAO["grid_eps"])),
        scale_base_mu=float(hp.get("scale_base_mu", KAN_PADRAO["scale_base_mu"])),
        scale_base_sigma=float(hp.get("scale_base_sigma", KAN_PADRAO["scale_base_sigma"])),
        sb_trainable=bool(hp.get("sb_trainable", True)),
        sp_trainable=bool(hp.get("sp_trainable", True)),
        symbolic_enabled=False,
        save_act=save_act,
        auto_save=False,
        seed=int(seed),
        device=DEVICE,
    )


def _tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


def _bce_finita(logits, targets):
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("A perda BCE da KAN não é finita.")
    return loss


def prever(model, X):
    model.eval()

    x = _tensor(X)

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "A entrada contém NaN ou Inf."
        )

    with torch.no_grad():
        logits = model(x).reshape(-1)

        if not torch.isfinite(logits).all():
            print("\n=== DIAGNÓSTICO KAN ===")
            print("Logits não finitos.")

            for nome, tensor in model.state_dict().items():
                if not torch.isfinite(tensor).all():
                    print(f"Tensor inválido: {nome}")
                    print("NaN:", torch.isnan(tensor).sum().item())
                    print("Inf:", torch.isinf(tensor).sum().item())

            print("Entrada:")
            print("Min:", x.min().item())
            print("Max:", x.max().item())
            print("Média:", x.mean().item())
            print("Desvio:", x.std().item())

            raise FloatingPointError(
                "A KAN produziu logits não finitos."
            )

        p = torch.sigmoid(logits).cpu().numpy()

    return p

def treinar(Xtr, ytr, hp, seed, Xval=None, yval=None):
    if (Xval is None) != (yval is None):
        raise ValueError("Xval e yval devem ser fornecidos juntos.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = criar_modelo(hp, np.asarray(Xtr).shape[1], seed)
    verificar_estado(model, "Criação")
    Xtr_t, ytr_t = _tensor(Xtr), _tensor(ytr).reshape(-1, 1)
    if Xval is None:
        Xval_t, yval_t = Xtr_t, ytr_t
    else:
        Xval_t, yval_t = _tensor(Xval), _tensor(yval).reshape(-1, 1)
    dataset = {
        "train_input": Xtr_t,
        "train_label": ytr_t,
        "test_input": Xval_t,
        "test_label": yval_t,
    }

    with torch.no_grad():
        logits = model(Xtr_t).reshape(-1)

        loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, 1),
            ytr_t
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("A perda inicial da KAN não é finita.")
        print("Loss inicial:", loss.item())
    
    model.train()
    model.zero_grad()

    logits = model(Xtr_t).reshape(-1, 1)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        ytr_t
    )

    loss.backward()

    model.zero_grad()

    os.makedirs(model.ckpt_path, exist_ok=True)
    history = model.fit(
        dataset,
        opt=_nome_otimizador(hp),
        steps=int(hp.get("steps", KAN_PADRAO["steps"])),
        log=10,
        lamb=float(hp.get("lamb", 0.0)),
        lamb_l1=float(hp.get("lamb_l1", 1.0)),
        lamb_entropy=float(hp.get("lamb_entropy", 2.0)),
        lamb_coef=float(hp.get("lamb_coef", 0.0)),
        lamb_coefdiff=float(hp.get("lamb_coefdiff", 0.0)),
        update_grid=bool(hp.get("update_grid", True)),
        grid_update_num=int(hp.get("grid_update_num", 10)),
        start_grid_update_step=int(hp.get("start_grid_update_step", -1)),
        stop_grid_update_step=int(hp.get("stop_grid_update_step", 50)),
        lr=float(hp.get("lr", 1.0)),
        batch=int(hp.get("batch", -1)),
        loss_fn=_bce_finita,
    )
    verificar_estado(model, "Após treinamento")
    verificar_historico(history, "Após treinamento")
    return model, history


def fit(hp, fold, Xlist, seed):
    Xtr, ytr, Xval, yval = fold
    model, _ = treinar(Xtr, ytr, hp, seed, Xval, yval)
    return [prever(model, X) for X in Xlist], prever(model, Xtr), int(hp.get("steps", KAN_PADRAO["steps"]))


def fit_refine(hp, fold, Xlist, seed, new_grid):
    Xtr, ytr, Xval, yval = fold
    if int(new_grid) <= int(hp.get("grid", KAN_PADRAO["grid"])):
        raise ValueError("A grade refinada deve ter mais intervalos que a grade original.")
    total_steps = int(hp.get("steps", KAN_PADRAO["steps"]))
    steps_before = max(1, total_steps // 2)
    steps_after = max(1, total_steps - steps_before)
    config_before = refinar_hp(hp, steps=steps_before)
    model, _ = treinar(Xtr, ytr, config_before, seed, Xval, yval)
    model = model.refine(int(new_grid))
    verificar_estado(model, "Após refinamento da grade")
    config = refinar_hp(hp, grid=int(new_grid), steps=steps_after)
    Xtr_t, ytr_t = _tensor(Xtr), _tensor(ytr).reshape(-1, 1)
    Xval_t, yval_t = _tensor(Xval), _tensor(yval).reshape(-1, 1)
    os.makedirs(model.ckpt_path, exist_ok=True)
    history = model.fit(
        {
            "train_input": Xtr_t,
            "train_label": ytr_t,
            "test_input": Xval_t,
            "test_label": yval_t,
        },
        opt=_nome_otimizador(config),
        steps=int(config.get("steps", KAN_PADRAO["steps"])),
        log=10,
        lamb=float(config.get("lamb", 0.0)),
        lamb_l1=float(config.get("lamb_l1", 1.0)),
        lamb_entropy=float(config.get("lamb_entropy", 2.0)),
        lamb_coef=float(config.get("lamb_coef", 0.0)),
        lamb_coefdiff=float(config.get("lamb_coefdiff", 0.0)),
        update_grid=bool(config.get("update_grid", True)),
        grid_update_num=int(config.get("grid_update_num", 10)),
        start_grid_update_step=int(config.get("start_grid_update_step", -1)),
        stop_grid_update_step=int(config.get("stop_grid_update_step", 50)),
        lr=float(config.get("lr", 1.0)),
        batch=int(config.get("batch", -1)),
        loss_fn=_bce_finita,
    )
    verificar_estado(model, "Após treinamento")
    verificar_historico(history, "Após treinamento refinado")
    return [prever(model, X) for X in Xlist], prever(model, Xtr), history


def fit_final(hp, final, steps, seed):
    Xtr, ytr, Xtest, _ = final
    config = copy.deepcopy(hp)
    if np.isscalar(steps):
        config["steps"] = max(1, int(steps))
    else:
        passos = np.asarray(steps, dtype=float)
        config["steps"] = max(1, int(np.median(passos))) if len(passos) else int(config.get("steps", KAN_PADRAO["steps"]))
    model, _ = treinar(Xtr, ytr, config, seed)
    return prever(model, Xtest)


def hp_kan(trial):
    arquitetura = trial.suggest_categorical("arquitetura", ["4", "8", "16", "8x8"])
    opt = trial.suggest_categorical("opt", ["LBFGS", "Adam"])
    lr = (
        trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True)
        if opt == "Adam"
        else trial.suggest_float("lr_lbfgs", 1e-2, 1.0, log=True)
    )
    return {
        **KAN_PADRAO,
        "hidden": tuple(int(x) for x in arquitetura.split("x")),
        "grid": trial.suggest_categorical("grid", [3, 5, 7]),
        "k": trial.suggest_categorical("k", [2, 3]),
        "base_fun": trial.suggest_categorical("base_fun", ["silu", "identity", "zero"]),
        "opt": opt,
        "lr": lr,
        "steps": trial.suggest_categorical("steps", [50, 100, 150]),
        "batch": trial.suggest_categorical("batch", [-1, 256, 512]),
        "grid_eps": trial.suggest_float("grid_eps", 0.0, 1.0),
    }


def hp_from_params(params):
    hp = copy.deepcopy(KAN_PADRAO)
    hp.update(params)
    arquitetura = hp.pop("arquitetura", "8")
    hp["hidden"] = tuple(int(x) for x in arquitetura.split("x"))
    if "lr_adam" in hp:
        hp["lr"] = hp.pop("lr_adam")
    elif "lr_lbfgs" in hp:
        hp["lr"] = hp.pop("lr_lbfgs")
    return hp


def carregar_dobras(features=(), over=True, path=comum.CAMINHO_CSV):
    """Cria as cinco dobras oficiais sem ajustar o pré-processamento na validação."""
    from sklearn.model_selection import StratifiedKFold

    Xtr, _, ytr, _ = comum.dividir(comum.carregar_df(path))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=comum.SEMENTE)
    folds = []
    for idx_tr, idx_val in skf.split(Xtr, ytr):
        Xt, yt, Xv = comum.preparar(
            Xtr.iloc[idx_tr], ytr.iloc[idx_tr], Xtr.iloc[idx_val],
            features=features, over=over,
        )
        folds.append(
            (
                Xt.to_numpy(dtype="float32"),
                np.asarray(yt, dtype="float32"),
                Xv.to_numpy(dtype="float32"),
                ytr.iloc[idx_val].to_numpy(dtype="float32"),
            )
        )
    return folds


def medir_config(hp, folds, seeds=(42,)):
    linhas = []
    for seed in seeds:
        for i, fold in enumerate(folds):
            Xval, yval = fold[2], fold[3]
            fold_seed = int(seed) + i
            try:
                (pval,), _, _ = fit(hp, fold, [Xval], fold_seed)
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"semente={fold_seed}, dobra={i}: {exc}"
                ) from exc
            linhas.append({"semente": int(seed), "dobra": i, "ks": comum.ks(yval, pval),
                           "auroc": comum.metricas(yval, pval, 0.5)["auroc"], "ce": comum.ce(yval, pval),
                           "brier": comum.mse(yval, pval)})
    kss = np.asarray([linha["ks"] for linha in linhas])
    return {
        "ks": float(kss.mean()),
        "ks_dp": float(kss.std(ddof=1)) if len(kss) > 1 else 0.0,
        "ks_dobras": kss.tolist(),
        "linhas": linhas,
    }


def comparar_bloco(base, variantes, folds, seeds=(42,)):
    referencia = medir_config(base, folds, seeds)
    saida = []
    for nome, hp in variantes.items():
        try:
            medida = medir_config(hp, folds, seeds)
        except FloatingPointError as exc:
            saida.append({
                "variante": nome,
                "status": "falhou",
                "ks": float("nan"),
                "ks_dp": float("nan"),
                "delta_ks": float("-inf"),
                "ganhou_dobras": 0,
                "empatou_dobras": 0,
                "perdeu_dobras": len(referencia["ks_dobras"]),
                "erro": str(exc),
                "config": hp,
            })
            continue
        delta = np.asarray(medida["ks_dobras"]) - np.asarray(referencia["ks_dobras"])
        saida.append({
            "variante": nome,
            "status": "ok",
            "ks": medida["ks"],
            "ks_dp": medida["ks_dp"],
            "delta_ks": float(delta.mean()),
            "ganhou_dobras": int(np.sum(delta > 0)),
            "empatou_dobras": int(np.sum(delta == 0)),
            "perdeu_dobras": int(np.sum(delta < 0)),
            "erro": "",
            "config": hp,
        })
    return referencia, saida


def refinar_hp(hp, **mudancas):
    novo = copy.deepcopy(hp)
    novo.update(mudancas)
    return novo


def avaliar_placar(hp, features=(), over=True, divs=None):
    """Mede o placar comum; seleciona passos só na partição interna de parada."""
    Xtr, _, ytr, _ = comum.dividir(comum.carregar_df())
    passos = []

    def ajustar_prever(X, y, Xval, ajuste, parada, seed):
        ajuste, parada = np.asarray(ajuste), np.asarray(parada)
        Xa, ya, Xp = comum.preparar(
            X.iloc[ajuste], y.iloc[ajuste], X.iloc[parada],
            features=features, over=over,
        )
        modelo_parada, historico = treinar(
            Xa.to_numpy(dtype="float32"), np.asarray(ya, dtype="float32"),
            hp, seed, Xp.to_numpy(dtype="float32"), y.iloc[parada].to_numpy(dtype="float32"),
        )
        perdas = np.asarray(historico["test_loss"], dtype=float)
        if not len(perdas) or not np.isfinite(perdas).all():
            raise FloatingPointError("A perda de parada da KAN é vazia ou não finita.")
        melhor_passo = int(np.argmin(perdas)) + 1
        passos.append(melhor_passo)

        Xfit, yfit, Xout = comum.preparar(X, y, Xval, features=features, over=over)
        config = refinar_hp(hp, steps=melhor_passo)
        modelo, _ = treinar(
            Xfit.to_numpy(dtype="float32"), np.asarray(yfit, dtype="float32"),
            config, seed,
        )
        return prever(modelo, Xout.to_numpy(dtype="float32"))

    medida = comum.medir_placar(Xtr, ytr, ajustar_prever, divs=divs)
    medida["steps_mediana"] = int(np.median(passos))
    medida["steps_por_dobra"] = passos
    return medida


def prever_teste(hp, features=(), over=True, steps=None, seed=42):
    """Treina a configuração congelada no treino oficial e só então prevê o teste."""
    Xtr, Xtest, ytr, _ = comum.dividir(comum.carregar_df())
    Xfit, yfit, Xout = comum.preparar(Xtr, ytr, Xtest, features=features, over=over)
    config = refinar_hp(hp, steps=int(steps or hp["steps"]))
    model, _ = treinar(
        Xfit.to_numpy(dtype="float32"), np.asarray(yfit, dtype="float32"),
        config, seed,
    )
    return prever(model, Xout.to_numpy(dtype="float32"))
