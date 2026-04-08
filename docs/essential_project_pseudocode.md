# Essential Project Pseudocode

This document turns the main implementation into high-level pseudocode.
It focuses on the parts that drive the full pipeline:

- data preparation
- heuristic score precomputation
- difficulty-aware negative sampling
- baseline training
- curriculum training
- HeaRT evaluation

The pseudocode below matches the current codebase rather than the earlier
design-only phase notes.

---

## 1. Prepare Link Prediction Data

**Main file:** `utils/data_utils.py`

```text
FUNCTION prepare_link_prediction_data(dataset_name, root, val_ratio, test_ratio, seed):
    dataset <- load_dataset(dataset_name, root)
    data <- dataset[0]

    pos_undirected_edges <- remove_self_loops_and_duplicate_undirected_edges(data.edge_index)

    shuffled_edges <- shuffle(pos_undirected_edges, seed)

    num_val <- floor(len(shuffled_edges) * val_ratio)
    num_test <- floor(len(shuffled_edges) * test_ratio)
    num_train <- len(shuffled_edges) - num_val - num_test

    train_pairs <- first num_train edges
    val_pairs <- next num_val edges
    test_pairs <- remaining edges

    train_pos_edge_index <- edge_index(train_pairs, undirected = false)
    val_pos_edge_index <- edge_index(val_pairs, undirected = false)
    test_pos_edge_index <- edge_index(test_pairs, undirected = false)

    train_edge_index <- edge_index(train_pairs, undirected = true)

    all_pos_edge_index <- unique_undirected_edges(data.edge_index)

    val_neg_edge_index <- sample_random_non_edges(
        excluding = all_pos_edge_index,
        count = number of validation positives,
        seed = seed + 1
    )
    test_neg_edge_index <- sample_random_non_edges(
        excluding = all_pos_edge_index,
        count = number of test positives,
        seed = seed + 2
    )

    RETURN dictionary with:
        raw graph data
        node features x
        train graph edges
        train/val/test positive edges
        val/test negative edges
        num_nodes
        num_features
```

Why it matters:
- Training uses only training positives inside the message-passing graph.
- Validation and test use fixed negatives so metrics are comparable.

---

## 2. Encode and Decode Edge Scores - 

**Main files:** `models/base.py`, `models/gcn.py`, `models/gat.py`

```text
FUNCTION model_forward(x, train_edge_index, edge_label_index):
    z <- encode node embeddings from graph structure
    scores <- decode edge_label_index using embeddings z
    RETURN scores
```

```text
FUNCTION encode(x, train_edge_index):
    FOR each GNN layer except the last:
        x <- graph_convolution(x, train_edge_index)
        x <- nonlinearity(x)
        x <- dropout(x)

    x <- final_graph_convolution(x, train_edge_index)
    RETURN x
```

```text
FUNCTION decode(z, edge_label_index):
    (src, dst) <- edge_label_index

    IF decoder is inner_product:
        RETURN rowwise_sum(z[src] * z[dst])

    IF decoder is hadamard_mlp:
        RETURN MLP(z[src] * z[dst])

    IF decoder is edge_mlp:
        RETURN MLP(concatenate(z[src], z[dst]))
```

Why it matters:
- Every training and evaluation path reduces to "encode node embeddings, then
  score positive and negative edges."

---

## 3. Precompute Difficulty Scores For Candidate Negatives - needed

**Main files:** `scripts/precompute_scores.py`, `negative_sampling/heuristics.py`

```text
FUNCTION precompute(dataset, heuristic, neg_ratio, seed, output_dir):
    split <- prepare_link_prediction_data(dataset, seed = seed)

    train_pos <- split.train_pos_edge_index
    train_graph <- split.train_edge_index
    num_nodes <- split.num_nodes

    n_candidates <- neg_ratio * number of train positives

    candidates <- sample_candidate_non_edges(
        num_nodes = num_nodes,
        excluding = train_pos,
        count = n_candidates,
        seed = seed
    )

    scores <- compute_heuristic_scores(
        edge_index = train_graph,
        num_nodes = num_nodes,
        negatives = candidates,
        heuristic = heuristic
    )

    save compressed file with:
        candidates
        scores
        train graph edge_index
        num_nodes
        heuristic name
```

```text
FUNCTION compute_heuristic_scores(edge_index, num_nodes, negatives, heuristic):
    G <- build undirected NetworkX graph from edge_index

    FOR each candidate pair (u, v) in negatives:
        IF heuristic is common_neighbors:
            score <- number of shared neighbors of u and v
        ELSE IF heuristic is adamic_adar:
            score <- sum over shared neighbors w of 1 / log(degree(w))
        ELSE IF heuristic is resource_allocation:
            score <- sum over shared neighbors w of 1 / degree(w)

    RETURN score array
```

Why it matters:
- Training does not compute heuristic scores on the fly.
- The saved candidate pool is what makes curriculum sampling fast.

---

## 4. Build Difficulty Buckets And Sample Negatives - needed 

**Main file:** `negative_sampling/sampler.py`

```text
CLASS DifficultyBasedSampler(candidates, scores, seed):
    sort candidate indices by score ascending

    split sorted indices into 3 quantile buckets:
        easy   <- bottom third
        medium <- middle third
        hard   <- top third

    store bucket indices
```

```text
FUNCTION sample_by_difficulty(n, bucket, epoch_offset):
    pool <- indices for selected bucket
    rng <- seeded random generator using base seed + epoch_offset

    IF n > bucket size:
        sample with replacement
    ELSE:
        sample without replacement

    RETURN selected candidate edges
```

```text
FUNCTION sample_mixed(n, weights, epoch_offset):
    normalize weights into [easy, medium, hard]
    counts <- rounded sample counts per bucket
    counts[last] <- n - sum(previous counts)

    sampled_parts <- empty list

    FOR each bucket in [easy, medium, hard]:
        IF requested count is 0:
            continue

        IF bucket is empty:
            sample from all candidates as fallback
        ELSE:
            sample from this bucket

        append sampled edges to sampled_parts

    concatenate sampled_parts along edge dimension
    RETURN sampled negative edges
```

Why it matters:
- Difficulty is implemented by score quantiles, not fixed thresholds.
- Curriculum learning changes only the bucket mixture, not the model code.

---

## 5. Baseline Training Loop

**Main file:** `experiments/train_baseline.py`

```text
FUNCTION train_epoch(model, data_dict, optimizer, device, neg_ratio, seed):
    set model to train mode
    zero optimizer gradients

    x <- node features
    train_edge_index <- graph used for message passing
    pos_edge_index <- train positive edges

    num_pos <- number of positive train edges
    num_neg <- num_pos * neg_ratio

    all_positive_edges <- all undirected positives from original graph
    neg_edge_index <- sample_random_non_edges(
        excluding = all_positive_edges,
        count = num_neg,
        seed = seed
    )

    z <- model.encode(x, train_edge_index)
    pos_logits <- model.decode(z, pos_edge_index)
    neg_logits <- model.decode(z, neg_edge_index)

    logits <- concatenate(pos_logits, neg_logits)
    labels <- [1 for positives, 0 for negatives]

    loss <- binary_cross_entropy_with_logits(logits, labels)
    backpropagate loss
    optimizer step

    RETURN loss
```

```text
FUNCTION main_baseline_training(args):
    set random seeds
    device <- cuda if available else cpu

    data_dict <- prepare_link_prediction_data(args.dataset, args.seed)
    model <- build GCN or GAT
    optimizer <- Adam(model.parameters, lr = args.lr)
    logger + checkpoint manager <- initialize

    FOR epoch from 1 to args.epochs:
        epoch_seed <- args.seed * 10000 + epoch
        loss <- train_epoch(..., seed = epoch_seed)

        IF epoch is evaluation step:
            val_metrics <- evaluate(model, validation split)
            save checkpoint keyed by validation AUC
            log epoch metrics

    load best checkpoint
    test_metrics <- evaluate(model, test split)

    IF HeaRT is enabled:
        heart_metrics <- run hard-negative evaluation

    save JSON results and epoch CSV
```

Why it matters:
- This is the control condition: random negatives during training.

---

## 6. Standard Evaluation Metrics

**Main files:** `experiments/train_baseline.py`, `utils/metrics.py`

```text
FUNCTION evaluate(model, data_dict, split, device):
    set model to eval mode

    x <- node features
    train_edge_index <- training graph
    pos_edge_index <- fixed positive edges for split
    neg_edge_index <- fixed negative edges for split

    z <- model.encode(x, train_edge_index)
    pos_scores <- model.decode(z, pos_edge_index)
    neg_scores <- model.decode(z, neg_edge_index)

    RETURN:
        AUC
        Average Precision
        MRR
        Hits@10
        Hits@50
        Hits@100
```

Why it matters:
- Baseline and curriculum models share the same standard evaluation path.

---

## 7. Curriculum Scheduler And Competence Tracking - needed

**Main files:** `curriculum/competence.py`, `curriculum/scheduler.py`

```text
CLASS CompetenceMeter(window_size, smoothing):
    history <- bounded queue of recent validation metrics
    ema_value <- optional running exponential average

FUNCTION update(metric_value):
    add metric_value to history
    update ema if needed

FUNCTION get_competence():
    IF no history:
        RETURN 0
    IF smoothing is ema:
        RETURN ema_value
    RETURN moving_average(history)

FUNCTION is_threshold_reached(threshold):
    RETURN get_competence() >= threshold

FUNCTION reset():
    clear history and ema state
```

```text
CLASS CurriculumScheduler(phases, adaptive, fixed_phase_epochs, competence_window):
    current_phase_idx <- 0
    phase_history <- [(0, first phase name)]
    competence_meter <- CompetenceMeter(competence_window)

FUNCTION step(metric_value, epoch):
    phase_changed <- false
    competence_meter.update(metric_value)

    IF adaptive mode:
        IF current phase threshold is reached:
            advance_phase(epoch)
        RETURN

    IF fixed schedule mode AND epoch is multiple of fixed_phase_epochs:
        advance_phase(epoch)

FUNCTION advance_phase(epoch):
    IF already in final phase:
        RETURN

    current_phase_idx <- current_phase_idx + 1
    phase_changed <- true
    append (epoch, new phase name) to phase_history
    competence_meter.reset()
```

Why it matters:
- The scheduler controls when the sampler should expose harder negatives.

---

## 8. Curriculum Training Loop - needed

**Main file:** `experiments/train_curriculum.py`

```text
FUNCTION train_epoch_with_negatives(model, data_dict, optimizer, neg_edge_index, device):
    set model to train mode
    zero optimizer gradients

    x <- node features
    train_edge_index <- training graph
    pos_edge_index <- train positive edges

    z <- model.encode(x, train_edge_index)
    pos_logits <- model.decode(z, pos_edge_index)
    neg_logits <- model.decode(z, neg_edge_index)

    logits <- concatenate(pos_logits, neg_logits)
    labels <- [1 for positives, 0 for negatives]

    loss <- binary_cross_entropy_with_logits(logits, labels)
    backpropagate loss
    optimizer step

    RETURN loss
```

```text
FUNCTION train_with_curriculum(
    model,
    data_dict,
    optimizer,
    scheduler,
    sampler,
    device,
    epochs,
    eval_every,
    neg_ratio
):
    num_neg_samples <- num_train_positives * neg_ratio

    FOR epoch from 1 to epochs:
        ratios <- scheduler.get_current_difficulty_ratios()

        neg_edges <- sampler.sample_mixed(
            n = num_neg_samples,
            weights = ratios,
            epoch_offset = epoch
        )

        loss <- train_epoch_with_negatives(
            model, data_dict, optimizer, neg_edges, device
        )

        IF epoch is evaluation step:
            val_metrics <- evaluate(model, validation split)
            scheduler.step(val_metrics["auc"], epoch)

            log:
                validation metrics
                current phase
                phase name
                whether phase changed
                competence value
                easy/medium/hard ratios

            save checkpoint using validation AUC

    load best checkpoint
    test_metrics <- evaluate(model, test split)

    RETURN test metrics + phase summary + training metadata
```

```text
FUNCTION main_curriculum_training(args):
    data_dict <- prepare_link_prediction_data(args.dataset, args.seed)

    candidates, scores <- load precomputed heuristic pool from disk
    sampler <- DifficultyBasedSampler(candidates, scores, seed = args.seed)

    phases <- build curriculum preset
    scheduler <- CurriculumScheduler(
        phases = phases,
        adaptive = args.adaptive,
        fixed_phase_epochs = args.fixed_phase_epochs,
        competence_window = args.competence_window
    )

    model <- build GCN or GAT
    optimizer <- Adam(...)

    results <- train_with_curriculum(...)

    IF HeaRT is enabled:
        heart_metrics <- run hard-negative evaluation

    save JSON results, phase history, bucket sizes, and epoch CSV
```

Why it matters:
- This is the main experiment: same model, but harder negatives are injected
  progressively through the sampler and scheduler.

---

## 9. HeaRT Evaluation - needed

**Main file:** `negative_sampling/heart.py`

```text
CLASS HeaRTEvaluator(data, heuristics, num_neg_per_pos, precomputed_dir, seed, dataset_name):
    all_positive_edges <- all edges in original graph
    heuristic_pools <- empty dictionary

    FOR each requested heuristic:
        load matching precomputed .npz file
        candidates <- stored negative candidate pairs
        scores <- stored heuristic scores

        build node_to_indices map:
            for each candidate pair (u, v):
                add candidate index to node_to_indices[u]
                add candidate index to node_to_indices[v]

        sort each node's candidate indices by score descending
        store candidates, scores, node_to_indices
```

```text
FUNCTION candidate_pairs_for_positive(u, v):
    merged_scores <- empty map from candidate pair to best score

    FOR each heuristic pool:
        indices <- candidates touching u plus candidates touching v

        FOR each candidate index:
            pair <- canonical undirected version of candidate edge
            merged_scores[pair] <- max(previous score, current heuristic score)

    RETURN candidate pairs sorted by merged score descending
```

```text
FUNCTION generate_test_set(pos_edge_index):
    hard_negs_per_pos <- empty list
    all_pos <- set of all graph positives

    FOR each positive edge (u, v):
        selected_pairs <- empty list
        seen_pairs <- empty set

        FOR pair in candidate_pairs_for_positive(u, v):
            IF pair is a true edge OR pair already selected:
                continue

            add pair to selected_pairs
            add pair to seen_pairs

            IF selected_pairs has num_neg_per_pos items:
                break

        IF selected_pairs has too few items:
            fallback random negatives <- sample remaining non-edges
            append fallback pairs until size reaches num_neg_per_pos

        convert selected_pairs to edge tensor
        append to hard_negs_per_pos

    RETURN pos_edge_index, hard_negs_per_pos
```

```text
FUNCTION evaluate_model(model, data_dict, device):
    set model to eval mode

    x <- node features
    train_edge_index <- training graph
    pos_edge_index <- test positives

    z <- model.encode(x, train_edge_index)
    (_, hard_negs_per_pos) <- generate_test_set(pos_edge_index)

    reciprocal_ranks <- []
    hits10 <- []
    hits50 <- []
    hits100 <- []

    FOR each positive test edge i:
        pos_score <- model score for positive edge i
        neg_scores <- model scores for paired hard negatives

        rank <- 1 + number of hard negatives scoring higher than pos_score

        append 1 / rank to reciprocal_ranks
        append rank <= 10 to hits10
        append rank <= 50 to hits50
        append rank <= 100 to hits100

    RETURN averages:
        heart_mrr
        heart_hits@10
        heart_hits@50
        heart_hits@100
```

Why it matters:
- HeaRT is the repository's "hard" evaluation protocol.
- It checks whether the model can beat structurally plausible non-edges, not
  just random negatives.

---

## 10. Full Pipeline At A Glance

```text
LOAD dataset
SPLIT graph edges into train / val / test
BUILD model

PRECOMPUTE heuristic scores for a large candidate negative pool
BUCKET candidates into easy / medium / hard

IF running baseline:
    train with fresh random negatives every epoch
ELSE IF running curriculum:
    use scheduler to choose bucket ratios
    sample negatives from easy / medium / hard mixture each epoch

SELECT best checkpoint by validation AUC
RUN standard test evaluation
OPTIONALLY RUN HeaRT hard-negative evaluation
SAVE metrics, logs, phase history, and summaries
```

