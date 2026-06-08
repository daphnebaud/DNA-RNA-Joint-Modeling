# Joint Modelling of DNA and RNA

This project leverages deep learning to decipher the regulatory code driving gene expression in *Saccharomyces cerevisiae*. By training models to reconstruct masked DNA and predict masked RNA expression, we aim to map influential cis-regulatory elements directly from the genomic sequence context.

## 📁 Dataset Structure
* **`data.npz`**: Contains the raw forward-strand DNA sequences (integer-encoded) and binary RNA expression tracks.
* **`regions.parquet`**: An index mapping continuous genomic regions, detailing offsets and chromosome contigs.
* **`ensembl_annotation.gff3`**: Provides external biological context and gene annotations.

## 🧠 Architectures
* **CNN (BPNet Variant):** A pooling-free, exponentially dilated residual network that preserves base-pair resolution to predict nucleotide identity and RNA expression.
* **Transformer (Encoder-Only):** A 4-layer multi-head self-attention network designed to capture global, long-range dependencies across the combined DNA/RNA input vectors.

## ⚙️ Training Methodology
We use a self-supervised, cross-modal masking strategy to force the models to learn complex regulatory grammar:
* **RNA Masking:** 50% of continuous expression labels within a window are masked and must be predicted from context.
* **DNA Masking:** 15% of nucleotides are hidden using a BERT-style masking approach (80% mask token, 10% random, 10% unchanged) and must be reconstructed.

## 📊 Key Findings
* **Performance:** Both models significantly outperformed random baselines. The CNN showed a slight advantage in RNA expression prediction, though reconstructing masked DNA proved intrinsically challenging for both.
* **Biological Validity:** Sequence logo analysis confirmed the models learned true biological motifs, particularly highlighting A/T-rich regions linked to polyadenylation and transcription termination.
* **Interpretability:** *In silico* mutagenesis successfully pinpointed highly influential regulatory clusters (e.g., the SDH7 promoter), validating the models' ability to map real regulatory impacts.

  
 Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: For specific PyTorch installations, please refer to the official PyTorch website to get the correct command for your system and CUDA version.)*
