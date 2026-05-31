"""Super Cérebro — Transformer causal unificado com Asset Embeddings.

Um único modelo que devora janelas de barras de TODOS os ativos. A identidade do
ativo entra por **embeddings** (ativo + classe) que condicionam cada token, de
modo que o backbone Transformer compartilhado aprende a física universal do fluxo
enquanto se especializa por ativo — o cross-asset learning pedido.

Blindagem de look-ahead:
  * cada amostra usa apenas a janela de barras PASSADAS até a barra-alvo;
  * o ``causal_mask`` do Transformer impede qualquer atenção a posições futuras;
  * o padding (início da série) é zero/finito — sem ``key_padding_mask`` para
    não criar linhas totalmente mascaradas (que produziriam NaN).

Este módulo importa ``torch`` no topo e por isso só deve ser carregado sob demanda
(ver ``research.models.get_super_brain``). Nasce pronto para GPU (usa CUDA quando
disponível); em CI roda um smoke test leve em CPU.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from innova_ea.research.dataset import PerAssetScaler, UniversalPanel, label_to_class
from innova_ea.research.models.base import UniversalModel


class _SuperBrainNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        n_asset_classes: int,
        *,
        window: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        n_out: int = 3,
    ) -> None:
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.asset_emb = nn.Embedding(n_assets, d_model)
        self.class_emb = nn.Embedding(n_asset_classes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_out),
        )
        causal = torch.triu(torch.full((window, window), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def forward(self, x, asset_id, class_id):
        # x: [B, L, F]; condiciona cada token na identidade do ativo.
        # O causal_mask já blinda 100% o look-ahead; a janela contém só barras
        # passadas e o padding (início da série) é zero (finito) — não usamos
        # key_padding_mask para evitar linhas totalmente mascaradas (→ NaN).
        h = self.input_proj(x) + self.pos
        cond = self.asset_emb(asset_id) + self.class_emb(class_id)  # [B, d]
        h = h + cond.unsqueeze(1)
        h = self.encoder(h, mask=self.causal_mask)
        last = self.norm(h[:, -1, :])
        return self.head(last)


class SuperBrain(UniversalModel):
    """Wrapper de treino/inferência do Transformer causal universal."""

    def __init__(
        self,
        panel: UniversalPanel,
        *,
        window: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        patience: int = 4,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(panel)
        self.window = window
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        d_ff=d_ff, dropout=dropout)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler = PerAssetScaler()
        self.net: _SuperBrainNet | None = None

    # ----------------------------------------------------- construção de janelas
    def _build_sequences(self, df: pl.DataFrame, *, with_labels: bool):
        """Monta janelas causais [N, L, F] por ativo, alinhadas às linhas de ``df``."""
        L, F = self.window, len(self.feature_names)
        df = df.with_row_index("_row")
        windows = np.zeros((df.height, L, F), dtype=np.float32)
        pad = np.ones((df.height, L), dtype=bool)        # True = posição ignorada
        asset_ids = np.zeros(df.height, dtype=np.int64)
        class_ids = np.zeros(df.height, dtype=np.int64)
        labels = np.zeros(df.height, dtype=np.int64)

        for (aid,), sub in df.sort(["asset_id", "time"]).group_by(["asset_id"], maintain_order=True):
            rows = sub["_row"].to_numpy()
            Xs = self.scaler.transform(sub).astype(np.float32)     # [Na, F] padronizado
            cid = int(sub["asset_class_id"][0])
            y = label_to_class(sub["label"].to_numpy()) if with_labels else None
            for i in range(sub.height):
                lo = max(0, i - L + 1)
                seg = Xs[lo : i + 1]
                r = rows[i]
                windows[r, L - seg.shape[0]:] = seg
                pad[r, L - seg.shape[0]:] = False
                asset_ids[r] = int(aid)
                class_ids[r] = cid
                if with_labels:
                    labels[r] = y[i]
        return windows, pad, asset_ids, class_ids, labels

    # ----------------------------------------------------------------- treino
    def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "SuperBrain":
        torch.manual_seed(self.seed)
        # Scaler ajustado só no treino (sem vazamento).
        self.scaler._features = self.feature_names
        self._fit_scaler(train_df)

        Xtr, ptr, atr, ctr, ytr = self._build_sequences(train_df, with_labels=True)
        self.net = _SuperBrainNet(
            len(self.feature_names), self.n_assets, self.n_asset_classes,
            window=self.window, **self.cfg,
        ).to(self.device)

        counts = np.bincount(ytr, minlength=3).astype(np.float64)
        cw = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=cw.to(self.device))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        tens = self._to_tensors(Xtr, ptr, atr, ctr, ytr)
        n = tens[0].shape[0]

        valid = None
        if valid_df is not None and valid_df.height:
            Xv, pv, av, cv, yv = self._build_sequences(valid_df, with_labels=True)
            valid = self._to_tensors(Xv, pv, av, cv, yv)

        best = float("inf")
        best_state = None
        bad = 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                xb, pb, ab, cb, yb = (t[idx].to(self.device) for t in tens)
                opt.zero_grad()
                loss = criterion(self.net(xb, ab, cb), yb)
                loss.backward()
                opt.step()

            if valid is not None:
                vloss = self._eval_loss(valid, criterion)
                if vloss < best - 1e-4:
                    best, bad = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _fit_scaler(self, train_df: pl.DataFrame) -> None:
        # Reaproveita PerAssetScaler.fit, que espera (panel, train_df); criamos um
        # painel mínimo carregando apenas os metadados necessários.
        from innova_ea.research.dataset import UniversalPanel
        dummy = UniversalPanel(train_df, self.feature_names, self.asset_vocab, self.class_vocab)
        self.scaler.fit(dummy, train_df)

    def _to_tensors(self, X, pad, a, c, y):
        return (
            torch.from_numpy(X),
            torch.from_numpy(pad),
            torch.from_numpy(a),
            torch.from_numpy(c),
            torch.from_numpy(y),
        )

    @torch.no_grad()
    def _eval_loss(self, tensors, criterion) -> float:
        self.net.eval()
        x, p, a, c, y = (t.to(self.device) for t in tensors)
        return float(criterion(self.net(x, a, c), y).item())

    # -------------------------------------------------------------- inferência
    # ------------------------------------------------------------- persistência
    def save(self, path) -> None:
        """Salva pesos + config + scaler + vocabulário para deploy."""
        from pathlib import Path

        if self.net is None:
            raise RuntimeError("modelo não treinado")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "super_brain",
                "state_dict": self.net.state_dict(),
                "window": self.window,
                "cfg": self.cfg,
                "feature_names": self.feature_names,
                "asset_vocab": self.asset_vocab,
                "class_vocab": self.class_vocab,
                "scaler": {
                    "mean": self.scaler._mean,
                    "std": self.scaler._std,
                    "global_mean": self.scaler._global_mean,
                    "global_std": self.scaler._global_std,
                    "features": self.scaler._features,
                },
            },
            str(p / "super_brain.pt"),
        )

    @classmethod
    def load(cls, path) -> "SuperBrain":
        """Recarrega o Super Cérebro salvo (para validação/execução)."""
        from pathlib import Path

        from innova_ea.research.dataset import UniversalPanel

        bundle = torch.load(str(Path(path) / "super_brain.pt"), map_location="cpu",
                            weights_only=False)
        panel = UniversalPanel(
            pl.DataFrame(), bundle["feature_names"],
            bundle["asset_vocab"], bundle["class_vocab"],
        )
        obj = cls(panel, window=bundle["window"], **bundle["cfg"])
        obj.net = _SuperBrainNet(
            len(obj.feature_names), obj.n_assets, obj.n_asset_classes,
            window=obj.window, **obj.cfg,
        ).to(obj.device)
        obj.net.load_state_dict(bundle["state_dict"])
        sc = bundle["scaler"]
        obj.scaler._mean = sc["mean"]
        obj.scaler._std = sc["std"]
        obj.scaler._global_mean = sc["global_mean"]
        obj.scaler._global_std = sc["global_std"]
        obj.scaler._features = sc["features"]
        return obj

    @torch.no_grad()
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("modelo não treinado")
        self.net.eval()
        X, pad, a, c, _ = self._build_sequences(df, with_labels=False)
        out = np.empty((df.height, 3), dtype=np.float64)
        for s in range(0, df.height, self.batch_size):
            e = min(s + self.batch_size, df.height)
            xb = torch.from_numpy(X[s:e]).to(self.device)
            pb = torch.from_numpy(pad[s:e]).to(self.device)
            ab = torch.from_numpy(a[s:e]).to(self.device)
            cb = torch.from_numpy(c[s:e]).to(self.device)
            logits = self.net(xb, ab, cb)
            out[s:e] = torch.softmax(logits, dim=1).cpu().numpy()
        return out
