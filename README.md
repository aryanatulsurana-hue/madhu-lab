# madhu-lab
# Predicting Protein Conformational Changes Upon Ligand Binding

This project focuses on modeling and predicting the structural changes that occur in proteins when they transition from their apo (unbound) to holo (ligand-bound) states. Understanding these conformational shifts is critical for rational drug design, as binding-induced structural rearrangements often govern binding affinity, allosteric regulation, and functional activation.

The approach taken here uses a neural network trained on curated protein-ligand structural datasets to learn how local structural environments influence residue-level movement. The model leverages coordinate data extracted from PDB structures and incorporates secondary structure context to predict the displacement of Cα atoms upon ligand binding.

---

## **Key Features**
- Extraction and preprocessing of apo–holo protein structural pairs.
- Binding-site–centric representation of protein environments.
- Secondary structure–aware residue categorization.
- Neural network architecture optimized to predict residue-specific movements.
- Evaluation pipeline using residue-wise error heatmaps and statistical metrics.

---

## **Background & Motivation**
This project was developed in **Prof. M. S. Madhusudhan’s Structural Biology Lab**, where work centered around understanding conformational dynamics in proteins. During this project, protein structural data processing pipelines were written in Python, and deep learning approaches were used to analyze structure-based features.

I also presented 2 posters which showcase this work, which are also attached here along with my reports.
---

## **Repository Structure**
