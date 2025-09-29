import json, os
try:
    from jsonschema import validate  # optional
except Exception:  # ModuleNotFoundError or other
    validate = None  # type: ignore

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    schema_path = os.path.join(os.path.dirname(path), "schema.json")
    if os.path.exists(schema_path) and validate is not None:
        with open(schema_path, "r", encoding="utf-8") as sf:
            schema = json.load(sf)
        validate(instance=cfg, schema=schema)
    elif os.path.exists(schema_path) and validate is None:
        # Schema present but jsonschema not installed; proceed without validation
        pass
    return cfg
