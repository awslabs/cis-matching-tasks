
set.seed(2)

library(tidyverse)
library(glue)
library(furrr)
library(here)
library(cli)
n_cores <- parallel::detectCores() - 4
plan(multisession, workers = n_cores)

source(here("cis-matching-tasks", "R-code", "utils.R"))

dataset <- "morph"


df <- read_csv(here("cis-matching-tasks", "data", dataset, "embeddings.csv")) %>%
    rename(id_num = id) %>%
    select(-`...1`)

# some images are duplicated (i.e., same image but with different names. For these images, the embeddings will coincide. So let's first clean the data)
l2norm <- (df %>% select(contains("feature")))^2 %>%
    rowSums() %>%
    sqrt()
df <- df[!duplicated(l2norm), ]



df <- df %>%
    group_by(id_num) %>%
    sample_n(pmin(10, n()), replace = FALSE) %>%
    filter(n() > 1) %>%
    ungroup()



all_ids <- df %>%
    distinct(id_num) %>%
    pull(id_num)
length(all_ids)

# train for CIs, test for target metric
train_ids <- sample(all_ids, size = floor(0.5 * length(all_ids)), replace = FALSE)
test_ids <- setdiff(all_ids, train_ids)

# estimate thresholds
folder_thresholds <- here("cis-matching-tasks", "results", "thresholds", dataset, "unbalanced")
if (length(list.files(folder_thresholds)) == 0) {
    target_fmr <- c(1e-4, 1e-3, 3e-3, 1e-2, 1e-1, 5e-1, 9e-1)
    target_fnmr <- c(1e-3, 1e-2, 3e-3, 1e-1)
    thresholds_estimates <- 1:5e2 %>%
        future_map(~ get_thresholds_for_targets_separate(
            df = df[df$id_num %in% test_ids, ],
            target_fmr = target_fmr,
            target_fnmr = target_fnmr,
            n_ids_fmr = ifelse(dataset == "morph", 2e3, 300),
            n_ids_fnmr = ifelse(dataset == "morph", 25e3, length(test_ids))
        ), .progress = TRUE, .options = furrr_options(seed = TRUE)) %>%
        bind_rows()
    thresholds_values <- thresholds_estimates %>%
        group_by(target, target_value) %>%
        summarise(
            threshold = mean(threshold, na.rm = TRUE),
            true_fmr = mean(fmr),
            true_fnmr = mean(fnmr)
        )
    thresholds_values %>%
        write_csv(here(folder_thresholds, glue("thresholds.csv")))
}


thresholds_values <- read_csv(here(folder_thresholds, glue("thresholds.csv")))
thresholds_values <- thresholds_values %>% filter((target == "fmr" & target_value %in% c(1e-4, 1e-3, 3e-3, 1e-2)) | (target == "fnmr" & target_value %in% c(1e-3, 1e-2, 3e-3, 1e-1)))
params <- tibble(n_ids = seq(5, 100) %>% rev())
alpha <- seq(0.01, 0.2, by = 0.01)
thresholds <- thresholds_values %>% pull(threshold)
params <- params %>%
    filter(n_ids == 50)

for (row in 1:nrow(params)) {
    cli_alert("Iteration {row} of {nrow(params)}")

    n_ids <- params[row, ]$n_ids

    n_rep <- 1e2
    if (n_ids == 50 | n_ids == 100) n_rep <- 1e3
    out <- 1:n_rep %>%
        future_map(~ get_metrics_split(
            df = df[df$id_num %in% sample(train_ids, size = n_ids, replace = FALSE), ],
            n_ids = n_ids,
            alpha = alpha,
            thresholds = thresholds
        ),
        .progress = TRUE, .options = furrr_options(seed = TRUE)
        )

    out %>%
        bind_rows() %>%
        inner_join(thresholds_values %>% select(target, target_value, threshold, true_fmr, true_fnmr), by = "threshold") %>%
        mutate(n_ids = n_ids) %>%
        write_csv(here("cis-matching-tasks", "results", "coverage_summary", dataset, "unbalanced", glue("n_ids={n_ids}_coverage_summary.csv")))
}


# pointise intervals for the ROC
source(here("cis-matching-tasks", "R-code", "utils_roc.R"))
target_fmr <- c(9e-1, 5e-1, 1e-2, 1e-3, 1e-4)
thresholds_values <- read_csv(here(folder_thresholds, glue("thresholds.csv")))
thresholds_values_sub <- thresholds_values %>%
    filter(target_value %in% target_fmr) %>%
    filter(target == "fmr")
params <- tibble(n_ids = seq(5, 100) %>% rev())
alpha <- seq(0.01, 0.2, by = 0.01)
thresholds <- thresholds_values_sub %>% pull(threshold)
params <- params %>% filter(n_ids == 50) # | n_ids != 100)


for (row in 1:nrow(params)) {
    cli_alert("Iteration {row} of {nrow(params)}")

    n_ids <- params[row, ]$n_ids

    n_rep <- 1e2
    if (n_ids == 50 | n_ids == 100) n_rep <- 1e3
    out <- 1:n_rep %>%
        future_map(~ get_metrics_split_fixed_fmr(
            df = df[df$id_num %in% sample(train_ids, size = n_ids, replace = FALSE), ],
            n_ids = n_ids,
            alpha = alpha,
            target_fmr = target_fmr,
            alpha_fmr = 0.1
        ), .progress = TRUE, .options = furrr_options(seed = TRUE))

    out %>%
        bind_rows() %>%
        inner_join(thresholds_values_sub %>% select(target, target_value, true_fnmr), by = c("target_fmr" = "target_value")) %>%
        mutate(n_ids = n_ids) %>%
        write_csv(here("cis-matching-tasks", "results", "coverage_summary", dataset, "unbalanced", glue("n_ids={n_ids}_coverage_summary_roc.csv")))
}
