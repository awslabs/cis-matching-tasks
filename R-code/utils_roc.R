

# methods ----

get_bootstrap_db <- function(G, target_fmr, smooth = FALSE) {
    # computes double-or-nothing bootstrap confidence intervals for FNMR @ FMR
    n_ids <- length(unique(G$id1))
    weights <- sample(c(0, 2), size = n_ids, replace = TRUE)

    ids_selected <- (1:n_ids)[weights == 2]

    G_genuine_scores <- G$genuine_scores[G$id1 %in% ids_selected] %>%
        na.omit() %>%
        as.vector()
    G_impostor_scores <- G$impostor_scores[(G$id1 %in% ids_selected) & (G$id2 %in% ids_selected)] %>%
        na.omit() %>%
        as.vector()

    roc_estimate <- pROC::roc(cases = G_genuine_scores, controls = G_impostor_scores, quiet = TRUE)
    if (smooth) {
        # alternative distributions
        # dist_genuine <- MASS::fitdistr(G_genuine_scores, "t", start = list(m = 1e-1, s = 1e-1, df = 10), lower = c(0, 1e-1, 1), upper = c(100, 100, 30))
        # dist_impostor <- MASS::fitdistr(G_impostor_scores, "t", start = list(m = 0.5, s = 1e-1, df = 10), lower = c(0, 1e-1, 1), upper = c(100, 100, 30))
        # dist_genuine <- MASS::fitdistr(G_genuine_scores, "cauchy", lower = c(0, 0))
        # dist_impostor <- MASS::fitdistr(G_impostor_scores, "cauchy", lower = c(0, 0))
        # dist_genuine <- MASS::fitdistr(G_genuine_scores, "log-normal")
        # dist_impostor <- MASS::fitdistr(G_impostor_scores, "log-normal")
        dist_genuine <- MASS::fitdistr(G_genuine_scores, "normal")
        dist_impostor <- MASS::fitdistr(G_impostor_scores, "normal")
        roc_estimate <- pROC::smooth(roc_estimate,
            method = "density",
            fit.cases = dist_genuine, fit.controls = dist_impostor
        )
    }

    est <- pROC::coords(roc_estimate, x = 1 - target_fmr, input = "specificity") %>%
        as_tibble() %>%
        mutate(fmr = 1 - specificity, fnmr = 1 - sensitivity) %>%
        dplyr::select(fnmr, fmr)
    fmr <- est %>% pull(fmr)
    fnmr <- est %>% pull(fnmr)

    return(tibble(fnmr = fnmr, fmr = fmr))
}


get_bootstrap_vertex <- function(G, target_fmr, smooth = FALSE) {
    # computes vertex bootstrap confidence intervals for FNMR @ FMR
    n_ids <- length(unique(G$gt))
    m <- sum(G$gt == 1)
    index <- sample(1:n_ids, size = n_ids, replace = TRUE)
    index <- index %>%
        map(~ ((.x - 1) * m + 1):(.x * m)) %>%
        unlist()

    G$genuine_mat_scores <- G$genuine_mat_scores[index, index]
    # extract diagonal
    genuine_scores <- c()
    for (i in 1:n_ids) {
        index_i <- ((i - 1) * m + 1):(i * m)
        genuine_scores_i <- G$genuine_mat_scores[index_i, index_i]
        genuine_scores <- c(genuine_scores, as.vector(genuine_scores_i[upper.tri(genuine_scores_i)]))
    }
    G_genuine_scores <- genuine_scores %>%
        na.omit() %>%
        as.vector()

    # here skipping the correction
    G$impostor_mat_scores <- G$impostor_mat_scores[index, index]
    G_impostor_scores <- G$impostor_mat_scores[upper.tri(G$impostor_mat_scores)] %>%
        as.vector() %>%
        na.omit() %>%
        as.vector()

    preds <- c(G_genuine_scores, G_impostor_scores) / max(c(G_genuine_scores, G_impostor_scores))
    response <- c(rep(1, times = length(G_genuine_scores)), rep(0, times = length(G_impostor_scores)))
    if (smooth) {
        roc_estimate <- pROC::smooth(pROC::roc(response = response, predictor = preds, quiet = TRUE), method = "binormal")
    } else {
        roc_estimate <- pROC::roc(response = response, predictor = preds, quiet = TRUE)
    }

    est <- pROC::coords(roc_estimate, x = 1 - target_fmr, input = "specificity") %>%
        as_tibble() %>%
        mutate(fmr = 1 - specificity, fnmr = 1 - sensitivity) %>%
        select(fnmr, fmr)
    fmr <- est %>% pull(fmr)
    fnmr <- est %>% pull(fnmr)

    return(tibble(fnmr = fnmr, fmr = fmr))
}



# other utils ----


get_coverage <- function(n_ids, m, d, distribution_id, sigma2between, sigma2within, alpha, target_fmr, alpha_fmr = 0.05) {
    # Computes confidence intervals on synthetic data.

    G <- gen_data_emb(n_ids = n_ids, m = m, d = d, distribution_id = distribution_id, sigma2within = sigma2within, sigma2between = sigma2between)

    out <- get_ci_for_fixed_fmr(G, n_ids, m, alpha, target_fmr, alpha_fmr)

    return(list(out = out, roc_curve = NA))
}

get_metrics_split_fixed_fmr <- function(df, n_ids, alpha, target_fmr, alpha_fmr = 0.5) {
    # Computes confidence intervals on real-world data.

    df <- df %>% arrange(id_num)
    m <- df %>%
        count(id_num) %>%
        pull(n)
    embeddings <- df %>%
        select(matches("feature_"))

    G <- get_adj_mat(embeddings, n_ids, m)
    cis <- get_ci_for_fixed_fmr(G, n_ids, m, alpha, target_fmr, alpha_fmr)

    cis
}


# this will only work for the balanced setting
get_ci_for_fixed_fmr <- function(G, n_ids, m, alpha, target_fmr, alpha_fmr = 0.05) {
    # compute confidence intervals for FNMR @ FMR
    n <- m * n_ids

    thresholds <- target_fmr %>%
        map_dfr(~ estimate_threshold_target_fmr(G, .x) %>%
            mutate(target = "fmr") %>%
            mutate(target_value = .x)) %>%
        pull(threshold)

    mat_acc <- thresholds %>% map(~ get_genuine_and_impostor_mat(G, .x))
    out <- mat_acc %>%
        map(~ compute_matrix_at_id_level(.x$genuine_mat_acc, .x$impostor_mat_acc, G$m))

    current_fnmr <- mat_acc %>%
        map(~ mean(.x$genuine_mat_acc, na.rm = TRUE)) %>%
        unlist()
    current_fmr <- mat_acc %>%
        map(~ mean(.x$impostor_mat_acc, na.rm = TRUE)) %>%
        unlist()


    # wilson
    wilson_ci <- c()
    for (i in 1:length(out)) {
        impostor_mat_acc <- mat_acc[[i]]$impostor_mat_acc
        genuine_mat_acc <- mat_acc[[i]]$genuine_mat_acc
        fmr_mat <- out[[i]]$fmr_mat
        fnmr_mat <- out[[i]]$fnmr_mat
        m_mat <- out[[i]]$m_mat

        # get upper and lower bounds for FMR
        sample_analogue_var <- get_sample_analogue_var(genuine_mat_acc = genuine_mat_acc, impostor_mat_acc = impostor_mat_acc, fnmr_mat = fnmr_mat, fmr_mat = fmr_mat, m = G$m, m_mat = m_mat)
        fmr_wilson_bounds <- alpha_fmr %>%
            map_dfr(~ get_wilson(
                G = impostor_mat_acc,
                alpha = .x,
                sample_size = sum(!is.na(impostor_mat_acc)) / 2, # n_ids,
                var_sa = sample_analogue_var$var_fmr
            ) %>% setNames(c("lb_fmr", "ub_fmr"))) %>%
            mutate(alpha = alpha_fmr)

        # compute the thresholds to obtain those FMR levels
        thresholds_bounds_fmr <- c(fmr_wilson_bounds$lb_fmr, fmr_wilson_bounds$ub_fmr) %>%
            map_dfr(~ estimate_threshold_target_fmr(G, .x) %>%
                mutate(target = "fmr") %>%
                mutate(target_value = .x)) %>%
            pull(threshold)

        mat_acc_at_fmr_bounds <- thresholds_bounds_fmr %>% map(~ get_genuine_and_impostor_mat(G, .x))
        out_at_fmr_bounds <- mat_acc_at_fmr_bounds %>%
            map(~ compute_matrix_at_id_level(.x$genuine_mat_acc, .x$impostor_mat_acc, G$m))

        current_fnmr_at_fnmr_bounds <- out_at_fmr_bounds %>%
            map(~ mean(.x$fnmr_mat * .x$m_mat, na.rm = TRUE)) %>%
            unlist()
        current_fmr_at_fmr_bounds <- out_at_fmr_bounds %>%
            map(~ mean(.x$fmr_mat * .x$m_mat, na.rm = TRUE)) %>%
            unlist()

        for (j in 1:length(current_fnmr_at_fnmr_bounds)) {
            impostor_mat_acc <- mat_acc_at_fmr_bounds[[j]]$impostor_mat_acc
            genuine_mat_acc <- mat_acc_at_fmr_bounds[[j]]$genuine_mat_acc
            fmr_mat <- out_at_fmr_bounds[[j]]$fmr_mat
            fnmr_mat <- out_at_fmr_bounds[[j]]$fnmr_mat
            m_mat <- out_at_fmr_bounds[[j]]$m_mat

            sample_analogue_var <- get_sample_analogue_var(genuine_mat_acc = genuine_mat_acc, impostor_mat_acc = impostor_mat_acc, fnmr_mat = fnmr_mat, fmr_mat = fmr_mat, m = G$m, m_mat = m_mat)

            fnmr_wilson_bounds_j <- alpha %>%
                map_dfr(~ get_wilson(
                    G = genuine_mat_acc,
                    alpha = .x,
                    sample_size = sum(!is.na(genuine_mat_acc)) / 2, # n_ids,
                    var_sa = sample_analogue_var$var_fnmr
                ) %>%
                    setNames(c("lb_fnmr", "ub_fnmr"))) %>%
                mutate(alpha = alpha)

            if (j == 1) {
                fnmr_wilson_bounds <- fnmr_wilson_bounds_j %>% select(ub_fnmr, alpha)
            } else {
                fnmr_wilson_bounds <- fnmr_wilson_bounds %>% inner_join(fnmr_wilson_bounds_j %>% select(lb_fnmr, alpha), by = "alpha")
            }
        }
        wilson_ci <- wilson_ci %>% bind_rows(fnmr_wilson_bounds %>% mutate(target_fmr = target_fmr[i]) %>% mutate(type = "wilson") %>% mutate(lb_fmr = fmr_wilson_bounds$lb_fmr) %>% mutate(ub_fmr = fmr_wilson_bounds$ub_fmr))
    }


    # bootstraps
    n_boot <- 1e3
    # for the double-or-nothing, transform matrices of scores to vectors
    genuine_scores <- G$genuine_mat_scores %>% as.vector()
    impostor_scores <- G$impostor_mat_scores %>% as.vector()
    gt <- G$gt
    id1 <- rep(gt, times = length(gt))
    id2 <- rep(gt, each = length(gt))

    G_boot <- list(genuine_scores = genuine_scores, impostor_scores = impostor_scores, id1 = id1, id2 = id2)

    out_db <- 1:n_boot %>%
        map_dfr(~ get_bootstrap_db(G = G_boot, target_fmr = target_fmr) %>%
            mutate(target_fmr = target_fmr))

    out_db_smooth <- 1:n_boot %>%
        map_dfr(~ get_bootstrap_db(G = G_boot, target_fmr = target_fmr, smooth = TRUE) %>%
            mutate(target_fmr = target_fmr))

    # out_vertex <- 1:n_boot %>%
    #     map_dfr(~ get_bootstrap_vertex(G = G, target_fmr = target_fmr, smooth = FALSE) %>%
    #         mutate(target_fmr = target_fmr))

    # out_vertex_smooth <- 1:n_boot %>%
    #     map_dfr(~ get_bootstrap_vertex(G = G, target_fmr = target_fmr, smooth = TRUE) %>%
    #         mutate(target_fmr = target_fmr))


    out_boot <- out_db %>%
        mutate(type = "db") %>%
        bind_rows(out_db_smooth %>% mutate(type = "db_smooth")) %>%
        # bind_rows(out_vertex %>% mutate(type = "vertex")) %>%
        # bind_rows(out_vertex_smooth %>% mutate(type = "vertex_smooth")) %>%
        group_by(type, target_fmr) %>%
        summarise(
            mean_fmr = mean(fmr, na.rm = TRUE),
            var_fmr = var(fmr, na.rm = TRUE),
            lb_fmr = quantile(fmr, alpha / 2, na.rm = TRUE),
            ub_fmr = quantile(fmr, 1 - alpha / 2, na.rm = TRUE),
            mean_fnmr = mean(fnmr, na.rm = TRUE),
            var_fnmr = var(fnmr, na.rm = TRUE),
            lb_fnmr = quantile(fnmr, alpha / 2, na.rm = TRUE),
            ub_fnmr = quantile(fnmr, 1 - alpha / 2, na.rm = TRUE),
            alpha = alpha,
            .groups = "keep"
        )

    wilson_ci %>%
        bind_rows(out_boot) %>%
        inner_join(tibble(current_fmr = current_fmr, current_fnmr = current_fnmr, target_fmr = target_fmr), by = "target_fmr") %>%
        return()
}
