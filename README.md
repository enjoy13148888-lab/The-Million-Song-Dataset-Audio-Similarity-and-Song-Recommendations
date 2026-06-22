# Million Song Dataset: Audio Similarity and Recommendations

This project uses Apache Spark to explore the Million Song Dataset (MSD), engineer large-scale audio feature pipelines, build genre classification models, and train a collaborative filtering recommender system based on implicit user feedback.

## Project Overview
The Million Song Dataset combines one million song metadata records, user listening histories (Taste Profile subset), genre labels (MAGD), and high-dimensional audio features extracted via digital signal processing. Handling tens of millions of interactions and ~12.5 GB of audio feature files requires distributed processing and careful memory management.

In this project, I built an analytical pipeline that:
* Profiles and cleans MSD metadata, genre labels, Taste Profile interactions, and audio feature datasets stored in cloud object storage.
* Dynamically constructs Spark schemas from ARFF-style attribute files and standardizes feature naming for downstream modeling.
* Merges and analyzes audio features to support binary and multiclass genre classification.
* Trains and evaluates an implicit Alternating Least Squares (ALS) model to recommend songs to users based on their listening history.

## Tech Stack and Skills
* **Languages:** Python
* **Big Data Framework:** Apache Spark (PySpark)
* **Data Engineering:** Dynamic schema construction from ARFF metadata, robust joins across multiple large tables, and repartitioning/caching strategies for scalability.
* **Machine Learning:** Binary and multiclass classification (Logistic Regression, Random Forest, Gradient Boosted Trees), hyperparameter tuning, and implicit-feedback collaborative filtering (ALS).
* **Analytics & Evaluation:** Feature correlation analysis, multicollinearity reduction, long-tail distribution analysis, and ranking metrics (Precision@10, NDCG, MAP).

## Key Technical Highlights
* **Dynamic Schema Generation:** Parsed lightweight ARFF-style attribute files to automatically build precise Spark `StructType` schemas. Mapped numerical fields to `DoubleType` and identifiers to `StringType`, bypassing the slow and unreliable `inferSchema` method.
* **Feature Engineering at Scale:** Renamed long, fragile DSP descriptors into compact, systematic identifiers (e.g., A000, L000) to manage high-dimensional data. Resolved schema inconsistencies (hidden spaces/quotes) to successfully inner join five large audio feature datasets into a single cached table of 993,828 tracks.
* **Audio Feature Analysis & Genre Classification:** Analyzed spectral, rhythm, timbral, and LPC features to identify and remove constant or highly correlated columns, reducing multicollinearity. Trained a binary "Rap vs. Others" classifier, achieving an AUROC of 0.94, and explored the challenges of extreme class imbalance in a multiclass genre classifier on ~420k labeled tracks.
* **Scalable Recommender System (ALS):** Processed 48,373,586 user–song interactions under strict memory constraints using 400 partitions and memory-and-disk persistence. Filtered long-tail sparsity to train an implicit ALS model on 661,103 users and 161,173 songs.
* **Modeling Strategy:** Developed strategies for aggressive hyperparameter tuning (e.g., optimizing learning rates for GBT, regularization for logistic regression) and k-fold cross-validation to stabilize performance on highly skewed data.

## How to Use
This repository serves as a portfolio demonstration of building an end-to-end, scalable machine learning and ETL pipeline on the Million Song Dataset. You can read the final report to understand the architectural design and results, or inspect the PySpark notebooks to see how I implemented dynamic schemas, engineered high-dimensional features, and built an implicit-feedback recommender system in a distributed environment.
