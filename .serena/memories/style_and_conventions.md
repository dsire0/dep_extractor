# Style and Conventions
- **Naming Conventions**: Standard PEP 8 (snake_case for variables/functions, PascalCase for classes). Internal helper functions are prefixed with `_`.
- **Exhaustive Documentation**: Every generated function, class, and logic block must include PEP-257 standard docstrings and exhaustive inline explanations.
- **Rolling Backup Architecture (Rule of 5)**: Before live file changes, generate sequential read-only backups (`.bkp1` to `.bkp5`). Oldest is purged if exceeding 5. 
- **Restoration Rules**: Rollbacks require an explicit doc-block injected detailing timestamp, rationale, and preventative constraints, followed by a `serena_write_memory` call to log the failure mode.
- **Cognitive Clarity**: Group logic spatially, avoid dense syntactic walls, utilize fail-safe operations (e.g. `read_file_safe`), and keep UI feedback fast (via rich console outputs).
- **Issue Tracking & Commits**: ALL tasks tracked via Beads (`bd`). Commits must be step-by-step, Atomic, and rigidly follow the Conventional Commits specification.