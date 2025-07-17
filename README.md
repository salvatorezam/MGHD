# MGHD
Repo for the project MGHD: A Universal Mamba-Graph Hybrid Decoder Architecture for Real-Time Quantum Error Correction

---


### Repo structure

- scratchpad: transitory repo to put random stuff in, to be used not to bloat the main one

- core: the actual training/testing tool

    - codes: ideally contains special qec codes implementations, although not needed for now

    - datasets: self-explanatory. Each dataset folder, other than the data itself, contains additional info. The actual data dir is kept compressed most of the times, and uncompressed when needed.

        - google_qec3v5: google dataset for surface code threshold experimentations
            - google_qec3v5_experiment_data: data dir
            - paper-data-configs.yml
            - google_dataset_analysis.ipynb: explanatory notebook for dataset processing

    - decoders: contains a folder for each of the benchmarked decoders (also mghd), whether those are nn-based (directory has name with prefix model-<name>) or 'regular' ones (directory as name with prefix rule-<name>). If the code of the tool does not need big edits (e.g., a ready-made library exists for it), then this contains a simple wrapper, to maintain consistency with the general structure

        - model-astra:
            - paper-algo-configs.yml: config info for the decoding algorithm  taken from the original paper describing the tool, bot 
            - astra: the tool itself

    - results: subdirectory for storing computed data of all kinds

        - experiments: contains experiment data gathered after tests

        - trainings: training logs, configs, final weights, etc. of mghd (and potentially other models, if they have to be re-trained)

    - main.py: main python interface for running trainings and experiments

    - trainer.py: module for training mghd (and potentially other modules)

    - tester.py: module for testing tools agains datasets and sims

    - analyzer.py: separate module for processing computed data and plotting

---


### Worflow

1. main.py is run through cli with the appropriate training or testing mode selected, along with config file specifications

2. training (testing) is instantiated and launched: while it runs, logs and results are stored in 'results' folder, specifically within 'trainings' ('experiments')

3. in a separate stage, done manually after training (test) is over, analyzer.py reads data from 'results', processes it and outputs analysis, as well as plots