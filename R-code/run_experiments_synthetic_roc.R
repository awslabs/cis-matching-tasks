
library(tidyverse)
library(here)
library(furrr)
library(glue)
library(cli)

n_cores <- parallel::detectCores() - 4
plan(multisession, workers = n_cores)

source(here("cis-matching-tasks", "R-code", "utils.R"))

dataset <- "synthetic"
setting <- "balanced"
plot_folder <- here("cis-matching-tasks", "results", "figures", dataset, setting)

params <- tibble(distribution_id = "exponential", d = 128, sigma2between = NA) %>%
    crossing(tibble(m = c(5, 10))) %>%
    crossing(tibble(n_ids = c(50, 100))) %>%
    crossing(tibble(sigma2within = c(0.8^2, 5)))
params <- params %>%
    filter(m == 5 & sigma2within == 5)


# compute threshold for FMR and FNMR
params_threshold <- params %>%
    mutate(m = 10, n_ids = 200) %>%
    distinct()
folder_thresholds <- here("cis-matching-tasks", "results", "thresholds", "synthetic", "balanced")
files_thresholds <- list.files(folder_thresholds)

seq_target_fmr <- c(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1)
seq_target_fnmr <- seq_target_fmr
for (row in 1:nrow(params_threshold)) {
    cli_alert("Estimating the thresholds. Iteration {row} of {nrow(params_threshold)}")

    # parameters
    d <- (params_threshold %>% pull(d))[row]
    m <- (params_threshold %>% pull(m))[row]
    sigma2between <- (params_threshold %>% pull(sigma2between))[row]
    sigma2within <- (params_threshold %>% pull(sigma2within))[row]
    distribution_id <- (params_threshold %>% pull(distribution_id))[row]
    n_ids <- (params_threshold %>% pull(n_ids))[row]

    name_file <- glue("n_ids={n_ids}_m={m}_d={d}_distribution_id={distribution_id}_sigma2between={sigma2between}_sigma2within={sigma2within}_thresholds")
    if (length(files_thresholds) > 0) if (sum(grepl(name_file, files_thresholds)) > 0) next


    thresholds_estimates <- 1:1e3 %>%
        future_map(~ get_thresholds_for_targets(n_ids, m, d, distribution_id, sigma2between, sigma2within, target_fmr = seq_target_fmr, target_fnmr = seq_target_fnmr),
            .progress = TRUE,
            .options = furrr_options(seed = TRUE)
        ) %>%
        bind_rows()

    thresholds_estimates %>%
        group_by(target, target_value) %>%
        summarise(
            threshold = mean(threshold, na.rm = TRUE),
            true_fmr = mean(fmr),
            true_fnmr = mean(fnmr)
        ) %>%
        mutate(n_ids = n_ids, m = m, d = d, distribution_id = distribution_id, sigma2between = sigma2between, sigma2within = sigma2within) %>%
        write_csv(here(folder_thresholds, glue("{name_file}.csv")))
}


# FNMR @ FMR intervals
thresholds_values <- read_csv(list.files(folder_thresholds, full.names = TRUE))
seq_target_fmr <- c(1e-2, 1e-1, 3e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1)
alpha <- seq(0.01, 0.2, by = 0.01)
source(here("cis-matching-tasks", "R-code", "utils_roc.R"))

params <- params %>%
    filter(n_ids == 50)

for (row in 1:nrow(params)) {
    cli_alert("Iteration {row} of {nrow(params)}")

    d <- (params %>% pull(d))[row]
    m <- (params %>% pull(m))[row]
    sigma2between <- (params %>% pull(sigma2between))[row]
    sigma2within <- (params %>% pull(sigma2within))[row]
    distribution_id <- (params %>% pull(distribution_id))[row]
    n_ids <- (params %>% pull(n_ids))[row]

    thresholds_values_sub <- thresholds_values %>%
        filter(
            d == d,
            distribution_id == distribution_id,
            is.na(sigma2between) == is.na(sigma2between) | sigma2between == sigma2between,
            sigma2within == sigma2within
        ) %>%
        filter((target_value %in% seq_target_fmr & target == "fmr")) %>%
        filter(n_ids == max(n_ids)) %>%
        filter(m == max(m)) %>%
        distinct()

    n_rep <- 1e2
    if (n_ids == 50 | n_ids == 100) n_rep <- 1e3

    out <- 1:n_rep %>%
        future_map(~ get_coverage(n_ids, m, d, distribution_id, sigma2between, sigma2within, alpha, seq_target_fmr, alpha_fmr = 0.9)$out, .progress = TRUE, .options = furrr_options(seed = TRUE))

    out %>%
        map_dfr(~.x) %>%
        inner_join(thresholds_values_sub %>% select(target, target_value, true_fnmr), by = c("target_fmr" = "target_value")) %>%
        bind_cols(params[row, ]) %>%
        write_csv(here("cis-matching-tasks", "results", "coverage_summary", "synthetic", "balanced", glue("n_ids={n_ids}_m={m}_d={d}_distribution_id={distribution_id}_sigma2between={sigma2between}_sigma2within={sigma2within}_coverage_roc.csv")))
}
