# L2C-Insert → MeTRA_Slot_NCO Integration Plan (prototype)

**Goal:** Run L2C-Insert (the NeurIPS 2025 insertion-based constructive NCO
solver) against the MeTRA_Slot_NCO interface — the `train.py` Lightning
pipeline and the `rl4co/models/zoo/*` packaging.

**Verdict (from exploration):** a new folder under `rl4co/models/zoo/` is **only
the last ~20%** of the work. L2C-Insert is a standalone POMO-style codebase
(plain `nn.Module`, its own env/trainer/tester, `.txt` HGS datasets, supervised
insertion labels). MeTRA_Slot_NCO is an rl4co fork where zoo models are
`LightningModule`s driven by `pl.Trainer.fit` over cached `.pt` TensorDict
datasets. The two do not share a single interface. Below is the concrete path.

---

## 1. The fundamental mismatches

| Concern | L2C-Insert (source) | MeTRA_Slot_NCO (target) | Bridging work |
|---|---|---|---|
| Training loop | custom `Trainer`, per-step autoregressive rollout, `loss = -prob.log().mean()` (**supervised** CE on insertion labels) | `LightningModule.training_step(td)` (REINFORCE or another CE per batch) | wrap L2C rollout + supervision into a Lightning `training_step` |
| Model class | `VRPModel = CVRP_Encoder + CVRP_Decoder`, `nn.Module`, `mode='train'/'test'` | must be POMO/AM subclass (`LightningModule`) | new model class in zoo |
| Data | `<data>/CVRP/vrp100_hgs_train_n1000000.txt` — instances + **HGS reference solutions** (labels) | `data/{method}/{env}{N}_{dist}_{split}.pt` — locs/depot/demand/capacity + sparse `d_ins_idx/val`, **no solution labels** (`SlotDataset.__getitem__` returns no label) | new dataset/loader that exposes insertion labels, or generate them on the fly |
| Env state | `VRPEnv` carries partial subtours, `remaining_capacity_each_subtour`, `abs_scatter_solu_1`, from-depot flags | `CVRPEnv`/`TSPEnv` (rl4co) carry only locs/demand/capacity + decoder step index | decoder must rebuild L2C's insertion state from rl4co `td` + a rollout index |
| Supervision source | label generated via `generate_label(...)` from reference solutions | no reference solutions in `.pt` | **decide source of labels** (see §3) |

---

## 2. Why simply adding a zoo folder is not enough

`train.py` hard-codes the surface your new model must satisfy:

```python
# train.py
MODEL_CLASSES = {"pomo": POMOSlot, "am": AMSlot}     # line 30-31 — must add your model
model_cls = MODEL_CLASSES[backbone]
model_kwargs = dict(
    env=env_obj, embed_dim=embed_dim, num_slots=num_slots,
    **v_cfg, proj_dim=proj_dim, slot_iters=slot_iters,
    lambda_init=lambda_init, lr_dual=lr_dual,
    normalize_target=..., symmetrize_target=...,
    ins_method=ins_method, problem=env,
    optimizer_kwargs={"lr": t_cfg["lr"]},
)
model = model_cls(**model_kwargs)                     # must accept ALL of these
trainer.fit(model, train_loader, val_loader)          # must be a LightningModule
```

`env_obj` is an rl4co `CVRPEnv`/`TSPEnv`, and `train_loader` yields **dicts**
with keys `locs[, depot, demand, capacity, d_ins_idx, d_ins_val]` — **no
solution labels**. So the L2C encoder/decoder cannot even be fed without a data
and env bridge.

---

## 3. The key design decision: where do the insertion labels come from?

L2C-Insert is *supervised* by reference HGS solutions. MeTRA's pipeline has
none. You have three options (in increasing order of effort / faithfulness):

1. **Runtime insertion labels (recommended first cut).** During `training_step`,
   build a valid CVRP tour per instance greedily / with a fast heuristic (or by
   incremental insertion in the rollout itself), then compute each step's
   insertion target as the **minimum-cost insertion position** of the next node
   — i.e. reuse the "construction"/"insertion" (`RCIC`) d_ins machinery that
   already exists in [`rl4co/data/insertion_cost.py`](rl4co/data/insertion_cost.py).
   This keeps training data purely synthetic (no HGS `.txt` needed) and matches
   MeTRA's data format. **Downside:** labels are heuristic, not HGS-optimal —
   it is L2C's *architecture*, not its *supervision signal*.

2. **HGS labels for MeTRA format.** Port L2C's `.txt` (instances + HGS
   solutions) into the `.pt` format by adding a `solutions`/`labels` field to
   `SlotDataset`. This is the most faithful reproduction of the paper but
   requires regenerating / relabeling data and a label-aware dataset.

3. **REINFORCE refactor.** Drop supervision entirely and train insertion policy
   with a reward baseline (like POMOSlot). Biggest change to L2C's model;
   likely to hurt the insertion-paradigm benefit the paper reports. Least
   faithful to L2C.

**Recommendation:** start with **Option 1** — it is the smallest, self-contained
integration and exercises the full pipeline; revisit Option 2 once it runs.

---

## 4. File-by-file plan

### A. Copy the L2C architecture into the zoo (new folder)

```
rl4co/models/zoo/l2c_insert/
    __init__.py        # from .model import L2CInsertModel; __all__ = [...]
    encoder.py         # port of CVRP_Encoder (VRPModel.py:108-136)
    decoder.py         # port of CVRP_Decoder + DecoderLayer (VRPModel.py:137-354,
                       #   utils2.py helpers: get_encoding, node_flag_tran_to_,
                       #   generate_label, ...)
    model.py           # NEW LightningModule wrapper (see B)
```

- Keep `encoder.py` / `decoder.py` **as close to the original as possible** so
  you can diff against upstream and re-import the pretrained checkpoint later.
- Copy the small helpers from `CVRP/Train/utils2.py` (they are self-contained
  torch ops). Do **not** copy the whole 1300-line file.

### B. `L2CInsertModel(LightningModule)` — the interface adapter

Wrap the L2C encoder/decoder and implement the surfaces `train.py` requires:

```python
class L2CInsertModel(pl.LightningModule):
    def __init__(self, env, embed_dim=128, num_slots=8,           # accept no-op kwargs
                 proj_dim=64, slot_iters=3, lambda_init=1.0, lr_dual=1e-3,
                 normalize_target=True, symmetrize_target=True,
                 ins_method="construction", problem="cvrp",
                 optimizer_kwargs=None, **pomo_kwargs):
        ...
        self.encoder = CVRP_Encoder(...)    # ported
        self.decoder = CVRP_Decoder(...)    # ported
    def forward(self, td):
        # rebuild L2C insertion state from td (see C), roll out, return probs
    def training_step(self, batch, batch_idx):
        # Option 1: build labels, compute CE, log "loss"/"reward"
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
    def _dummy_custom_step(self, td): ...   # only if you keep metric/aux losses
```

- **Ignore** `num_slots` / `proj_dim` / `slot_iters` / `lambda_init` / `lr_dual`
  unless you keep the slot+metric machinery (the slot abstraction is a MeTRA
  addition; L2C itself has no slots). Keep the kwargs accepted so
  `train.py`'s `model_kwargs` don't crash.
- Register it: add to `MODEL_CLASSES` in [`train.py:30-31`](train.py#L30) (e.g.
  `"l2c": L2CInsertModel`) and to
  [`rl4co/models/zoo/__init__.py`](rl4co/models/zoo/__init__.py).
- `train_loader` yields **dicts**, not TensorDicts (`_collate_fn` in train.py).
  Convert to a TensorDict inside `training_step` (or a small helper) so the
  rl4co-idiomatic td access works.

### C. Env / state bridge (the real work)

L2C's decoder needs per-step *insertion* state: current partial solution,
`remaining_capacity_each_subtour`, `abs_scatter_solu_1` (selected/unselected),
from-depot flags. rl4co's `CVRPEnv` gives you locs/demand/capacity + a step
index only.

- Add a `render`/step helper that, given `td` and the current partial tour,
  computes the L2C state tensors. Reuse the *math* in
  [`L2C_Insert/CVRP/Train/VRPEnv.py`](../L2C_Insert/L2C_Insert/CVRP/Train/VRPEnv.py)
  (the subtour/capacity bookkeeping), but source inputs from the rl4co td.
- If you inherit from `POMO`/`AM` for the rollout, note their decoder expects a
  different state layout than L2C's — you likely **should not** inherit; write
  a standalone `LightningModule` and reuse only the rl4co `CVRPEnv` data
  (locs/demand/capacity) to build L2C's state.

### D. Evaluation (optional next step)

- Port `CVRP/Test/Tester.py`'s greedy insertion rollout as a `@torch.no_grad()`
  eval path, or reuse MeTRA's `scripts/eval_*.py` once the model produces
  standard `(B, N+1, 2)`/action tensors.
- For fidelity, load the upstream pretrained `.pt`
  (`L2C_Insert/CVRP/Test/result/pretrain/cvrp_model.pt`) — this requires your
  ported `encoder`/`decoder` to keep identical parameter names/shapes.

---

## 5. Order of implementation (prototype)

1. **Port encoder + decoder** with identical param names (enables later
   checkpoint loading). Smoke-test forward on a random td.
2. **Write the env/state bridge** — produce the L2C tensor inputs from an rl4co
   td (start TSP or CVRP; CVRP has subtour/capacity complexity).
3. **Wire `L2CInsertModel` + `training_step`** (Option-1 labels) with
   `configure_optimizers`.
4. **Register** in `MODEL_CLASSES` + zoo `__init__`, run `train.py` on small
   `N`/batch and hash it runs.
5. **Compare** against the `pomo`/`am` slot backbones on the same `N`/`dist`
   val data before trusting quality.

> Do **not** start the zoo folder until step 3 — the folder's content (the
> `LightningModule`) is the *output* of the state-bridge work, and its
> correctness depends entirely on steps 1-2.

---

## 6. Files touched (summary)

| File | Change |
|---|---|
| `rl4co/models/zoo/l2c_insert/__init__.py` | new — exports `L2CInsertModel` |
| `rl4co/models/zoo/l2c_insert/encoder.py` | new — ported `CVRP_Encoder` |
| `rl4co/models/zoo/l2c_insert/decoder.py` | new — ported `CVRP_Decoder` + helpers |
| `rl4co/models/zoo/l2c_insert/model.py` | new — `L2CInsertModel(LightningModule)` + rollout/state bridge |
| `rl4co/models/zoo/__init__.py` | add `L2CInsertModel` import |
| `train.py` | add `"l2c": L2CInsertModel` to `MODEL_CLASSES` (+ maybe a `--variant`-free branch) |
| `rl4co/data/insertion_cost.py` | (already present) Option-1 label source |
| (Option 2 only) `train.py SlotDataset` + generator | add `solutions`/`labels` field |
