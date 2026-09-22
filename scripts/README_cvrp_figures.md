# CVRP figures in MARS_plot

Run from the `MARS_plot` root. The plotter draws complete depot-to-depot
routes, customer demand, per-vehicle load and true Euclidean distance. It
checks that every customer appears exactly once and that every route fits the
vehicle capacity. Save to `.png` for slides or `.pdf` for a vector figure.

For a LEHD checkpoint and a LEHD-format `.txt` dataset with HGS routes:

```bash
python scripts/plot_cvrp_routes.py \
  --lehd-data /path/to/vrp100_hgs_validation.txt \
  --checkpoint /path/to/lehd.ckpt \
  --index 0 --device cpu --output figures/lehd_vs_hgs.png
```

The LEHD decode follows this repository's validation convention: the first
customer is anchored to the reference, then the model greedily selects the
remaining customers. It also records any depot return forced by insufficient
capacity; the newer LEHD environment repairs the teacher flag internally but
does not update its stored student flag. The figure verifies capacity again
before saving. Omit `--checkpoint` for a reference-only figure.

For any other MARS_plot backbone, export the instance and its decoded actions
to JSON. Node `0` is the depot; a `0` action separates vehicle routes:

```json
{
  "title": "MARS on CVRP100",
  "coordinates": [[0.1, 0.2], [0.5, 0.7], [0.9, 0.4]],
  "demands": [0, 2, 3],
  "capacity": 5,
  "solutions": [
    {"label": "MARS", "actions": [1, 0, 2]},
    {"label": "Reference", "routes": [[2], [1]]}
  ]
}
```

```bash
python scripts/plot_cvrp_routes.py --input-json instance.json \
  --output figures/comparison.pdf
```

One or two solutions are accepted. Each solution contains either `actions` or
`routes`; each route excludes the depot. For RL4CO's CVRP environment,
`td["locs"]` after reset includes the depot at index 0, while `td["demand"]`
has customers only: prepend zero before export. If demands are normalized,
use capacity `1`, or restore both demands and capacity to their original
units. Plotting only works for backbones that expose their decoded actions or
routes; a scalar reward alone cannot reconstruct a tour.
