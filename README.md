# Movie Recommender

This project turns a movie rating prediction assignment into a command-line Top-N movie recommender. The main recommendation model is Neural Collaborative Filtering trained with BPR ranking loss. Traditional models such as popularity ranking, bias baselines, and TruncatedSVD are kept as benchmarks.

The current recommendation workflow supports existing users only.

## Project Status

This is a learning and portfolio project, not a production recommendation service. It focuses on model training, ranking evaluation, baseline comparison, and command-line recommendation workflows.

Raw datasets, processed CSV files, trained checkpoints, caches, and generated reports are not included in the repository.

## Current Results

The fixed benchmark evaluates all models on the same eligible held-out positive rows. Ratings greater than or equal to `4.0` are treated as positive items.

Sample benchmark with `--max-users 1000` and `--svd-recall-size 1000`:

| model | HitRate@10 | NDCG@10 | HitRate@50 | NDCG@50 | HitRate@100 | NDCG@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PopularityBaseline | 0.0070 | 0.0034 | 0.0330 | 0.0089 | 0.0551 | 0.0125 |
| TruncatedSVDModel | 0.0060 | 0.0040 | 0.0290 | 0.0089 | 0.0591 | 0.0138 |
| NeuralCollaborativeFiltering | 0.0180 | 0.0080 | 0.0601 | 0.0168 | 0.1021 | 0.0236 |
| NeuralCFWithSVDRecall | 0.0200 | 0.0091 | 0.0641 | 0.0184 | 0.1101 | 0.0258 |

Neural CF with SVD candidate recall is the current main recommendation path because it performs best on the ranking benchmark.

A larger `TruncatedSVDModel` vs `NeuralCollaborativeFiltering` comparison also showed Neural CF ahead at `K=100`:

| model | HitRate@100 | Precision@100 | NDCG@100 |
| --- | ---: | ---: | ---: |
| TruncatedSVDModel | 0.0591 | 0.0006 | 0.0138 |
| NeuralCollaborativeFiltering | 0.1020 | 0.0010 | 0.0236 |

## Main Workflow

```bash
python train.py --force-process
python -m src.deep_learning.prepare_neural_cf_data
python -m src.deep_learning.train_neural_cf --epochs 20 --batch-size 1024 --patience 3
python -m src.recommend --user-id 1488844 --top-n 5
python -m src.benchmark_models --top-k 10 50 100 --max-users 1000 --svd-recall-size 1000
```

After processed CSV files and the Neural CF cache exist, reruns can usually start from training or recommendation.

## Features

- Parse raw movie rating data into train/test CSV files.
- Prepare reusable Neural CF interaction data for faster full-data training.
- Train Neural Collaborative Filtering with user/movie embeddings, an MLP scorer, BPR loss, early stopping, and configurable negative sampling.
- Recommend Top-N movies for an existing user with SVD candidate recall and Neural CF reranking.
- Compare Neural CF against Popularity and TruncatedSVD on fixed ranking benchmarks.
- Keep traditional RMSE baselines: `GlobalMeanModel`, `BiasModel`, and `TruncatedSVDModel`.
- Export recommendations and benchmark results to CSV.
- Inspect an existing user's highest-rated movies.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── reports/
├── src/
│   ├── benchmark_models.py
│   ├── compare_models.py
│   ├── config.py
│   ├── data_processing.py
│   ├── deep_learning/
│   │   ├── dataset.py
│   │   ├── evaluate_neural_cf.py
│   │   ├── model.py
│   │   ├── prepare_neural_cf_data.py
│   │   ├── recommend_neural_cf.py
│   │   └── train_neural_cf.py
│   ├── experiments.py
│   ├── model_store.py
│   ├── models.py
│   ├── ranking_metrics.py
│   ├── recommend.py
│   └── user_profile.py
├── train.py
├── requirements.txt
└── README.md
```

Generated data, trained checkpoints, and reports are ignored by Git.

## Data Setup

Raw data is not included in the repository. Place the files here:

```text
data/raw/data.txt
data/raw/movieTitles.csv
```

Generated files are saved under:

```text
data/processed/
models/
reports/
```

These generated paths are ignored by Git. Recreate them locally with the commands below.

## Install Dependencies

```bash
pip install -r requirements.txt
```

On the current development machine, dependencies are available through Anaconda `python`.

PyTorch installation can vary by operating system and hardware. If the generic `torch` dependency does not install cleanly, follow the official PyTorch installation command for your environment, then install the remaining requirements.

## Prepare Ratings

Create processed train/test CSV files from the raw ratings file:

```bash
python train.py --force-process
```

The current preprocessing entry point is `train.py`. It also trains the traditional RMSE baseline models and saves the TruncatedSVD baseline checkpoint. After the first run, `python train.py` reuses existing processed CSV files.

## Prepare Neural CF Data

Build the reusable Neural CF interaction cache before full training:

```bash
python -m src.deep_learning.prepare_neural_cf_data
```

The cache is saved to:

```text
data/processed/neural_cf_interactions.pkl
```

The prepare command scans the processed training CSV once, builds user/movie mappings, positive interactions, and rated-item sets, then saves them for reuse. Training reuses this cache by default when it matches the processed training CSV and Neural CF data options.

Rebuild the cache explicitly:

```bash
python -m src.deep_learning.prepare_neural_cf_data --force
```

## Train Main Model

Train the Neural CF main model:

```bash
python -m src.deep_learning.train_neural_cf --epochs 20 --batch-size 1024 --patience 3
```

Current training defaults use `embedding_dim=64` and `hidden_dims=[128, 64]`, selected from a small validation/BPR and ranking benchmark sweep as a balanced next-training structure. Existing checkpoints still load their saved architecture metadata.

A practical full-cache sampled run:

```bash
python -m src.deep_learning.train_neural_cf --max-positive-samples 200000 --epochs 5 --batch-size 4096 --patience 2
```

Train with mixed popularity-aware negative sampling:

```bash
python -m src.deep_learning.train_neural_cf \
  --epochs 5 \
  --batch-size 4096 \
  --patience 2 \
  --negative-sampling-strategy mixed \
  --mixed-negative-probability 0.5
```

Available negative samplers are `random`, `popularity`, and `mixed`.

The default Neural CF checkpoint is saved to:

```text
models/neural_cf_model.pt
```

For a small smoke test:

```bash
python -m src.deep_learning.train_neural_cf --max-rows 50000 --epochs 3 --batch-size 512
```

Smoke-test runs with `--max-rows` skip the default full-data cache unless you pass a custom `--data-cache-path`.

## Recommend Movies

Recommend movies with the Neural CF main model:

```bash
python -m src.recommend --user-id 1488844 --top-n 5
```

By default, `src.recommend` uses TruncatedSVD to recall candidate movies, then reranks those candidates with Neural CF. Tune the recall size:

```bash
python -m src.recommend --user-id 1488844 --top-n 5 --candidate-count 1000
```

Use full-catalog Neural CF scoring instead of SVD recall:

```bash
python -m src.recommend --user-id 1488844 --top-n 5 --candidate-source full
```

Save recommendations to CSV:

```bash
python -m src.recommend \
  --user-id 1488844 \
  --top-n 10 \
  --output reports/neural_cf_recommendations_user_1488844.csv
```

The Neural CF score is a ranking score, not a calibrated 1-5 star rating.

## Benchmark Models

Run the fixed benchmark with multiple K values:

```bash
python -m src.benchmark_models \
  --top-k 10 50 100 \
  --max-users 1000 \
  --svd-recall-size 1000 \
  --output reports/fixed_benchmark.csv
```

The benchmark evaluates `PopularityBaseline`, `TruncatedSVDModel`, `NeuralCollaborativeFiltering`, and `NeuralCFWithSVDRecall` on the same eligible held-out positive rows.

Evaluate one Neural CF checkpoint directly:

```bash
python -m src.deep_learning.evaluate_neural_cf --top-k 100 --max-users 100
```

Evaluate a custom checkpoint:

```bash
python -m src.deep_learning.evaluate_neural_cf --model-path /private/tmp/neural_cf_smoke.pt --top-k 100 --max-users 100
```

## Traditional Baselines

Train and evaluate traditional rating-prediction baselines:

```bash
python train.py
```

This prints RMSE for:

- `GlobalMeanModel`
- `BiasModel`
- `TruncatedSVDModel`

The TruncatedSVD baseline checkpoint is saved to:

```text
models/truncated_svd_model.pkl
```

Recommend movies with the TruncatedSVD baseline:

```bash
python -m src.recommend --model svd --user-id 1488844 --top-n 5
```

Compare different `TruncatedSVD` component sizes:

```bash
python -m src.experiments --n-components 10 20 50 --top-k 10 --max-users 100
```

For a faster RMSE-only SVD experiment:

```bash
python -m src.experiments --n-components 10 20 50 --skip-ranking
```

The older side-by-side SVD/Neural CF comparison command is still available:

```bash
python -m src.compare_models --top-k 100 --max-users 100
```

Prefer `src.benchmark_models` for current reports because it also includes the Popularity baseline and supports multiple K values in one run.

## Inspect a User

```bash
python -m src.user_profile --user-id 1488844 --top-n 10
```

This shows the user's rating count, average rating, rating range, and highest-rated movies.

## Verification

Run a syntax check:

```bash
python -m py_compile train.py src/*.py src/deep_learning/*.py
```

Run the main recommender after a Neural CF checkpoint exists:

```bash
python -m src.recommend --user-id 1488844 --top-n 5
```

Run a small benchmark:

```bash
python -m src.benchmark_models --top-k 10 50 100 --max-users 100 --svd-recall-size 1000
```

## Current Limitations

- Recommendations support existing users only.
- Cold-start recommendations are planned for a later phase.
- Ranking metrics currently evaluate one held-out positive item per test row.
