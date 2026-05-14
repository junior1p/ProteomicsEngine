# ProteomicsEngine

Pure Python DDA proteomics analysis pipeline.

## Features
- Peptide hyperscore scoring (Andromeda-style)
- Protein inference (parsimony)
- MaxLFQ-style label-free quantification
- Differential abundance (t-test + BH FDR)
- GO term enrichment (hypergeometric)

## Usage
```bash
pip install numpy scipy pandas matplotlib
python proteomics_engine.py
```

## Results (synthetic DDA, 6 samples, 1200 proteins)
- 8432 PSMs, 5757 passing filter
- 1190 proteins identified
- 97 differential proteins (FDR<0.05, |log2FC|>1)
