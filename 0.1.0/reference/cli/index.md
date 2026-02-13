# CLI Reference

## Installation

```
uv add "hypabase[cli]"
```

## Global options

| Option      | Default       | Description                      |
| ----------- | ------------- | -------------------------------- |
| `--db PATH` | `hypabase.db` | Path to the SQLite database file |

## Commands

### `init`

Initialize a new Hypabase database.

```
hypabase init
hypabase --db custom.db init
```

Creates the database file with the Hypabase schema. No-op if the file already exists.

### `node`

Create or update a node.

```
hypabase node ID [OPTIONS]
```

| Option         | Default   | Description               |
| -------------- | --------- | ------------------------- |
| `--type TEXT`  | `unknown` | Node type                 |
| `--props TEXT` | `None`    | JSON string of properties |

**Examples:**

```
hypabase node dr_smith --type doctor
hypabase node dr_smith --type doctor --props '{"specialty": "neurology"}'
```

### `edge`

Create a hyperedge connecting two or more nodes.

```
hypabase edge NODE1 NODE2 [NODE3 ...] [OPTIONS]
```

| Option               | Default      | Description                |
| -------------------- | ------------ | -------------------------- |
| `--type TEXT`        | *(required)* | Edge type                  |
| `--source TEXT`      | `None`       | Provenance source          |
| `--confidence FLOAT` | `None`       | Confidence score (0.0-1.0) |
| `--props TEXT`       | `None`       | JSON string of properties  |

**Examples:**

```
hypabase edge dr_smith patient_123 aspirin --type treatment
hypabase edge a b c --type link --source extraction --confidence 0.9
hypabase edge a b --type rel --props '{"weight": 0.5}'
```

### `query`

Query edges in the hypergraph.

```
hypabase query [OPTIONS]
```

| Option              | Default        | Description                                    |
| ------------------- | -------------- | ---------------------------------------------- |
| `--containing TEXT` | *(repeatable)* | Filter by node ID                              |
| `--type TEXT`       | `None`         | Filter by edge type                            |
| `--match-all`       | `False`        | Require all `--containing` nodes to be present |

**Examples:**

```
hypabase query --containing patient_123
hypabase query --containing patient_123 --containing aspirin --match-all
hypabase query --type treatment
```

### `stats`

Show database statistics: node and edge counts by type.

```
hypabase stats
```

### `validate`

Check internal consistency of the hypergraph.

```
hypabase validate
```

### `export-hif`

Export the hypergraph to HIF (Hypergraph Interchange Format) JSON.

```
hypabase export-hif OUTPUT_PATH
```

### `import-hif`

Import a hypergraph from HIF JSON.

```
hypabase import-hif INPUT_PATH
hypabase --db target.db import-hif INPUT_PATH
```
