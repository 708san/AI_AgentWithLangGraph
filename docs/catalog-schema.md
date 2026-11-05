# Project catalog schema

`project_ops inventory` produces a metadata-first JSON catalog. The default output is
`local_artifacts/catalog/project_catalog.json`, which is intentionally outside Git.

## Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Catalog format version |
| `catalog_type` | Always `project_inventory` for this catalog |
| `generated_at` | UTC generation time |
| `repository_root` | Absolute repository path |
| `git` | Branch, commit, dirty state, and tracked-file information |
| `summary` | File, script, data, artifact-root, category, and extension counts |
| `artifact_roots` | Summary of ignored/generated result roots |
| `scripts` | Python execution-script metadata |
| `data_files` | Data and derived-data metadata |
| `files` | Source and configuration file records |

## Safety rules

- Credentials and common `.env` files are excluded.
- Large ignored result directories are summarized as `artifact_roots`, not expanded into the source catalog.
- The catalog does not import application modules or call external APIs.
- Hashes are optional and can be limited by `--max-hash-mb`.
- With `--hash`, source, configuration, test, workflow, and registered data files are hashed; experimental and generated result files remain metadata-only to avoid scanning large transient trees.
- The catalog is descriptive evidence, not a replacement for the run manifest that will be introduced later.

## Examples

```bash
python -m project_ops inventory
python -m project_ops inventory --hash --output local_artifacts/catalog/project_catalog_hashed.json
```

