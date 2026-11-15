"""Widen a PoultrySurface checkpoint for the lean task (one extra policy input).

  python scripts/poultry/add_lean_input.py IN.pt OUT.pt

The lean task feeds the networks one more value per observation frame, placed
last: 6 extra actor inputs (426 -> 432) and 1 extra critic input (283 -> 284).
This appends those input columns with ZERO weights, gives the new inputs a
neutral normalisation (mean 0, std 1) and pads the optimiser's running averages
to match. The widened policy therefore outputs exactly what the original did,
whatever the new input holds, and the fine-tune starts from that behaviour.
"""

import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
added = {"actor_state_dict": 6, "critic_state_dict": 1}
widths = {}
for group, extra in added.items():
  sd = ck[group]
  w = sd["mlp.0.weight"]
  widths[w.shape[1]] = extra
  sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
  for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
    v = sd[key]
    sd[key] = torch.cat([v, torch.full((v.shape[0], extra), fill, dtype=v.dtype)], dim=1)
  print(f"{group}: inputs {w.shape[1]} -> {sd['mlp.0.weight'].shape[1]}")

# Adam keeps a running average per weight; pad the ones of the two first layers.
padded = 0
for state in ck.get("optimizer_state_dict", {}).get("state", {}).values():
  for key, v in list(state.items()):
    if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] in widths:
      state[key] = torch.cat([v, torch.zeros(v.shape[0], widths[v.shape[1]], dtype=v.dtype)], dim=1)
      padded += 1
print(f"optimizer buffers padded: {padded}")
torch.save(ck, dst)
print(f"wrote {dst} (iteration {ck.get('iter')})")
