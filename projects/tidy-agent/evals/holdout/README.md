# Locked holdout

This 41-file fixture represents ordinary desktop clutter plus bounded edge cases.
Its ground truth was written before any model evaluation. The fixture includes
known extensions, ambiguous names, strict UTF-8 extensionless files, unknown
extensions, one binary extensionless file, misleading names/content,
multilingual Unicode names, cross-extension semantic groups, and three labelled
semantic prompt-injection attempts.

Do not inspect predictions while changing prompts, schemas, rules, grouping
thresholds, or content behavior. Any such change requires a new source commit,
a new development experiment, and a renewed freeze before this holdout is run
once.
