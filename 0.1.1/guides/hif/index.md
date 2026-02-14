# HIF Import/Export

HIF (Hypergraph Interchange Format) is a JSON format for representing hypergraphs. Hypabase supports full round-trip import and export.

## Export

### Python API

```
hb = Hypabase("myproject.db")
hif_data = hb.to_hif()

# Write to file
import json
with open("export.json", "w") as f:
    json.dump(hif_data, f, indent=2)
```

### CLI

```
hypabase export-hif export.json
```

## Import

### Python API

```
import json

with open("export.json") as f:
    hif_data = json.load(f)

hb = Hypabase.from_hif(hif_data)

# The imported graph is in-memory. To persist:
# Option 1: Work with it in-memory
edges = hb.edges()

# Option 2: Save to a new database
# (use the storage layer directly for this)
```

### CLI

```
hypabase --db imported.db import-hif export.json
```

## HIF format structure

The HIF JSON contains nodes and edges with their full metadata:

```
{
  "nodes": [
    {
      "id": "dr_smith",
      "type": "doctor",
      "properties": {"specialty": "neurology"}
    }
  ],
  "edges": [
    {
      "id": "edge_uuid",
      "type": "treatment",
      "incidences": [
        {"node_id": "dr_smith", "direction": null},
        {"node_id": "patient_123", "direction": null}
      ],
      "source": "clinical_records",
      "confidence": 0.95,
      "properties": {}
    }
  ]
}
```

## Use cases

- **Backup and restore** — export a database, archive it, import it later
- **Migration** — move data between Hypabase instances
- **Sharing** — exchange hypergraph datasets with collaborators
- **Testing** — create fixtures from HIF files
- **Interop** — bridge to other tools that support HIF
