# Secondary methods for supplementary S3A–B.
# Input detail: eligible HPLC measurements joined to the analyte mapping.
# Input visible_predictions: portion ID, batch, position, grade and OOF visible score.
suppressPackageStartupMessages(library(dplyr))

heterogeneity_summary <- function(detail, bootstrap_replicates = 10000L, seed = 20260801L) {
  set.seed(seed)
eligible_detail <- detail %>%
  filter(strict_retained_range_eligible, short != "PGG", !is.na(observed_mg_g))

global_scales <- eligible_detail %>%
  group_by(compound_cn, compound, short, plot_order) %>%
  summarise(
    global_iqr_mg_g = IQR(observed_mg_g),
    global_median_mg_g = median(observed_mg_g),
    .groups = "drop"
  )

heterogeneity_batch <- eligible_detail %>%
  group_by(compound_cn, compound, short, plot_order, sample_number) %>%
  summarise(
    n_eligible_portions = n(),
    within_batch_range_mg_g = max(observed_mg_g) - min(observed_mg_g),
    batch_median_mg_g = median(observed_mg_g),
    .groups = "drop"
  ) %>%
  filter(n_eligible_portions >= 2L) %>%
  left_join(global_scales, by = c("compound_cn", "compound", "short", "plot_order")) %>%
  mutate(
    normalized_range_global_iqr = within_batch_range_mg_g / global_iqr_mg_g,
    relative_range_batch_median = if_else(
      batch_median_mg_g == 0,
      NA_real_,
      within_batch_range_mg_g / batch_median_mg_g
    ),
    complete_triplet = n_eligible_portions == 3L
  )

bootstrap_median <- function(values, replicates) {
  values <- values[is.finite(values)]
  if (length(values) < 2L) return(rep(NA_real_, replicates))
  replicate(replicates, median(sample(values, length(values), replace = TRUE)))
}

heterogeneity_summaries <- lapply(split(heterogeneity_batch, heterogeneity_batch$short), function(x) {
  boot_normalized <- bootstrap_median(x$normalized_range_global_iqr, bootstrap_replicates)
  boot_raw <- bootstrap_median(x$within_batch_range_mg_g, bootstrap_replicates)
  tibble(
    compound_cn = x$compound_cn[[1]],
    compound = x$compound[[1]],
    short = x$short[[1]],
    plot_order = x$plot_order[[1]],
    n_batches_at_least_two = nrow(x),
    n_complete_triplets = sum(x$complete_triplet),
    median_normalized_range_global_iqr = median(x$normalized_range_global_iqr, na.rm = TRUE),
    q1_normalized_range_global_iqr = quantile(x$normalized_range_global_iqr, 0.25, na.rm = TRUE),
    q3_normalized_range_global_iqr = quantile(x$normalized_range_global_iqr, 0.75, na.rm = TRUE),
    ci_low_normalized_range_global_iqr = quantile(boot_normalized, 0.025, na.rm = TRUE),
    ci_high_normalized_range_global_iqr = quantile(boot_normalized, 0.975, na.rm = TRUE),
    median_within_batch_range_mg_g = median(x$within_batch_range_mg_g, na.rm = TRUE),
    ci_low_within_batch_range_mg_g = quantile(boot_raw, 0.025, na.rm = TRUE),
    ci_high_within_batch_range_mg_g = quantile(boot_raw, 0.975, na.rm = TRUE),
    bootstrap_replicates = bootstrap_replicates
  )
}) %>%
  bind_rows() %>%
  arrange(plot_order)


  list(by_batch = heterogeneity_batch, summary = heterogeneity_summaries)
}

appearance_associations <- function(detail, visible_predictions, bootstrap_replicates = 10000L, permutation_replicates = 10000L) {
  # Call after heterogeneity_summary to continue the manuscript's RNG stream.
  eligible_detail <- detail %>% filter(strict_retained_range_eligible, short != "PGG", !is.na(observed_mg_g))
  compound_map <- detail %>% distinct(compound_cn, compound, short, plot_order) %>% arrange(plot_order)
association_frame <- eligible_detail %>%
  select(
    canonical_id,
    sample_number,
    group_index,
    compound_cn,
    compound,
    short,
    plot_order,
    observed_mg_g
  ) %>%
  inner_join(visible_predictions, by = c("canonical_id", "sample_number", "group_index"))

cluster_bootstrap_rho <- function(frame, x_name, replicates) {
  batches <- sort(unique(frame$sample_number))
  replicate(replicates, {
    sampled_batches <- sample(batches, length(batches), replace = TRUE)
    rows <- lapply(sampled_batches, function(batch) frame[frame$sample_number == batch, , drop = FALSE])
    boot <- bind_rows(rows)
    suppressWarnings(cor(boot[[x_name]], boot$observed_mg_g, method = "spearman"))
  })
}

batch_permutation_rho <- function(frame, x_name, replicates) {
  batches <- sort(unique(frame$sample_number))
  x_lookup <- visible_predictions %>%
    select(sample_number, group_index, all_of(x_name)) %>%
    mutate(key = paste(sample_number, group_index, sep = "::"))
  x_values <- setNames(x_lookup[[x_name]], x_lookup$key)
  original_batch_index <- match(frame$sample_number, batches)
  replicate(replicates, {
    permuted_batches <- sample(batches, length(batches), replace = FALSE)
    source_batches <- permuted_batches[original_batch_index]
    keys <- paste(source_batches, frame$group_index, sep = "::")
    suppressWarnings(cor(unname(x_values[keys]), frame$observed_mg_g, method = "spearman"))
  })
}

anchor_map <- c(
  expert_appearance_grade = "Recorded expert appearance grade",
  cross_fitted_visible_score = "Cross-fitted visible appearance score"
)

association_results <- list()
association_bootstrap <- list()
association_permutation <- list()
result_index <- 0L

for (anchor in names(anchor_map)) {
  for (endpoint in compound_map$short[1:7]) {
    frame <- association_frame %>% filter(short == endpoint)
    observed_rho <- suppressWarnings(
      cor(frame[[anchor]], frame$observed_mg_g, method = "spearman")
    )
    boot <- cluster_bootstrap_rho(frame, anchor, bootstrap_replicates)
    perm <- batch_permutation_rho(frame, anchor, permutation_replicates)
    p_value <- (1 + sum(abs(perm) >= abs(observed_rho), na.rm = TRUE)) /
      (1 + sum(is.finite(perm)))
    result_index <- result_index + 1L
    association_results[[result_index]] <- tibble(
      anchor_id = anchor,
      anchor = anchor_map[[anchor]],
      compound_cn = frame$compound_cn[[1]],
      compound = frame$compound[[1]],
      short = endpoint,
      plot_order = frame$plot_order[[1]],
      n_portions = nrow(frame),
      n_batches = n_distinct(frame$sample_number),
      spearman_rho = observed_rho,
      ci_low = quantile(boot, 0.025, na.rm = TRUE),
      ci_high = quantile(boot, 0.975, na.rm = TRUE),
      permutation_p = p_value,
      bootstrap_replicates = bootstrap_replicates,
      permutation_replicates = permutation_replicates
    )
    association_bootstrap[[result_index]] <- tibble(
      anchor_id = anchor,
      short = endpoint,
      replicate = seq_along(boot),
      spearman_rho = boot
    )
    association_permutation[[result_index]] <- tibble(
      anchor_id = anchor,
      short = endpoint,
      replicate = seq_along(perm),
      spearman_rho = perm
    )
  }
}

association_results <- bind_rows(association_results) %>%
  group_by(anchor_id) %>%
  mutate(q_bh = p.adjust(permutation_p, method = "BH")) %>%
  ungroup() %>%
  arrange(anchor_id, plot_order)


  association_results
}

secondary_analysis <- function(detail, visible_predictions, bootstrap_replicates = 10000L, permutation_replicates = 10000L, seed = 20260801L) {
  heterogeneity <- heterogeneity_summary(detail, bootstrap_replicates, seed)
  associations <- appearance_associations(detail, visible_predictions, bootstrap_replicates, permutation_replicates)
  list(heterogeneity = heterogeneity, associations = associations)
}
