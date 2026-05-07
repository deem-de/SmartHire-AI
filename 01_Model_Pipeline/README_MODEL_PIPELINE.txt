Model Pipeline Folder
=====================

This folder contains the core machine learning pipeline used in SmartHire.

Included parts
--------------
1. Data preparation and loading
2. Text preprocessing and BIO conversion
3. Section-aware splitting
4. BERT model definition
5. Training loop
6. Evaluation code
7. Final prepared training dataset
8. Final trained model outputs and saved metrics

Main files
----------
- main.py: main training and comparison entry point
- preprocessing.py: cleaning, BIO conversion, chunking, section splitting
- training.py: fine-tuning loop
- evaluation.py: token-level and entity-level evaluation
- feature_engineering.py: section helper logic
- data_loading.py: dataset loading utilities
- models.py: BERT model definition
- prepare_new_dataset.py: dataset preparation script
- analyze_pipeline.py: helper analysis script

Data
----
- data\resume_ner_training_data.json: final prepared training dataset

Trained outputs
---------------
- trained_models\baseline_model
- trained_models\section_aware_model
- trained_models\metrics.json
