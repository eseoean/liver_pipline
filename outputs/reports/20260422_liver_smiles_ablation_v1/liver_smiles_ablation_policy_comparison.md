# Liver SMILES policy comparison

| Split | Better policy | Drop-valid baseline | Keep-invalid zero | Delta |
|---|---:|---:|---:|---:|
| drug_group_4fold | drop invalid/no-SMILES | 0.5225 | 0.4163 | +0.1063 |
| random_4fold | drop invalid/no-SMILES | 0.8502 | 0.8056 | +0.0446 |
| scaffold_group_4fold | drop invalid/no-SMILES | 0.5145 | 0.4082 | +0.1063 |

Conclusion: liver does not reproduce the HNSC no-drop/zero-SMILES improvement; the valid-SMILES-only baseline remains stronger.
