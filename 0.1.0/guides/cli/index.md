# CLI Quickstart

Build a knowledge graph from the command line — no Python needed.

## Install

```
uv add "hypabase[cli]"
```

## Build a graph in five commands

Start with an empty database and populate it step by step:

```
# 1. Initialize the database
hypabase init
# Initialized Hypabase database at hypabase.db

# 2. Create nodes
hypabase node dr_smith --type doctor
hypabase node patient_123 --type patient
hypabase node aspirin --type medication

# 3. Create a hyperedge connecting all three
hypabase edge dr_smith patient_123 aspirin --type treatment --source clinical --confidence 0.95

# 4. Query edges containing a node
hypabase query --containing patient_123

# 5. Check database stats
hypabase stats
# Nodes: 3  Edges: 1
```

## Work with a specific database file

All commands default to `hypabase.db`. Use `--db` to target a different file:

```
hypabase --db research.db init
hypabase --db research.db node paper_1 --type paper
hypabase --db research.db edge paper_1 transformer bert --type builds_on
hypabase --db research.db stats
```

## Query with filters

Combine flags to narrow results:

```
# Edges containing both nodes
hypabase query --containing patient_123 --containing aspirin --match-all

# Edges of a specific type
hypabase query --type treatment
```

## Export and import

Move hypergraphs between databases using HIF (Hypergraph Interchange Format):

```
hypabase export-hif backup.json
hypabase --db copy.db import-hif backup.json
```

## Validate consistency

Check that the database has no orphaned references:

```
hypabase validate
# Hypergraph is valid.
```

See the [CLI Reference](https://hypabase.app/docs/latest/reference/cli/index.md) for all commands, flags, and options.
