# Submissions

A submission is to **one field**, as a Hugging Face model repository at a pinned revision holding
one directory per skill *of that field* (`spec.json` `model.layout`). Enter one field or both;
they have separate queues, separate kings and separate crowns, and a repository for one is not a
repository for the other.

```
pick_and_place/model.safetensors   the weights of a bpp_libero_v1 policy (see arch/)
pick_and_place/config.yaml         produced by `icilval convert-ckpt --arch-name bpp_libero_v1`
draw_anything/model.safetensors    the weights of a bpp_draw_v1 policy
draw_anything/config.yaml          produced by `icilval convert-ckpt --arch-name bpp_draw_v1`
```

Each `config.yaml`'s `model` block must equal that skill's architecture template except for
`model.mutable_keys` from `spec.json`. Optionally `README.md`, `.json`, `.txt`, `.yaml` files.
Anything else (in particular `.ckpt` / pickles) is ignored on download and rejected if it is the
only weights file. A repository missing one of its field's skill directories is refused: a duel
runs every skill of the field it is in. No participant code runs anywhere.

Queue it against the field it is for. With more than one field, saying which is **required** -
queueing against the wrong ladder is not something you could see from the reply.

## Converting BPP training checkpoints

```bash
icilval convert-ckpt --ckpt path/to/libero.ckpt --arch-name bpp_libero_v1 --out ./submission/pick_and_place
icilval convert-ckpt --ckpt path/to/draw.ckpt   --arch-name bpp_draw_v1   --out ./submission/draw_anything
icilval check-model ./submission
huggingface-cli upload <owner>/<name> ./submission
```

Then queue `owner/name@<commit sha>` through the organizer form or `icilval queue add`. The
converter also rewrites `_target_` paths from BPP's old package name (`umi_day.`) to
`behavior_prompting.`, which some public checkpoints still carry.

## What the validator checks

1. Allow-listed file extensions and total size (`model.max_repo_bytes`) over the whole repo.
2. Per skill: `config.yaml` parses with `yaml.safe_load`; `architecture == skills.<skill>.architecture`;
   every `_target_` appears in the template; every other value equals the template except
   `mutable_keys`.
3. Per skill: `model.safetensors` header: exactly the template's tensor names and shapes; dtypes in
   `model.allowed_dtypes`; parameter count ≤ `model.max_params`.
4. Weights load strictly into the template architecture inside a container with no network, one
   skill at a time.

The check is itemised per skill in the duel log and in the failure message the organizer form
shows.
