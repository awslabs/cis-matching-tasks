

# methods ------

get_subsetboot <- function(fnmr_mat, fmr_mat) {
  # computes subsets bootstrap estimates of FNMR and FMR
  # fnmr_mat is a diagonal matrix with FNMR values per identity
  # fmr_mat is a matrix with FMR values per identity pair
  if (!is.list(fmr_mat)) fmr_mat <- list(fmr_mat)
  if (!is.list(fnmr_mat)) fnmr_mat <- list(fnmr_mat)

  n_ids <- nrow(fmr_mat[[1]])
  ids_sampled <- sample(1:n_ids, size = n_ids, replace = TRUE)

  # FNMR
  fnmr_boot <- fnmr_mat %>%
    map(~ diag(.x)[ids_sampled[1:(n_ids - 1)]] %>% mean()) %>%
    unlist()

  # FMR
  fmr_boot <- fmr_mat %>%
    map(~ mean(.x[ids_sampled, ], na.rm = TRUE)) %>%
    unlist()

  return(tibble(fnmr = fnmr_boot, fmr = fmr_boot))
}

get_bootstrap_db <- function(fnmr_mat, fmr_mat, m_mat) {
  # computes double-or-nothing bootstrap estimates of FNMR and FMR
  # fnmr_mat is a diagonal matrix with FNMR values per identity
  # fmr_mat is a matrix with FMR values per identity pair
  # m_mat is a matrix with the number of samples per identity pair
  if (!is.list(fmr_mat)) fmr_mat <- list(fmr_mat)
  if (!is.list(fnmr_mat)) fnmr_mat <- list(fnmr_mat)

  n_ids <- nrow(fnmr_mat[[1]])
  weights <- sample(c(0, 2), size = n_ids, replace = TRUE)
  weights_mat <- matrix(weights, ncol = 1) %*% matrix(weights, nrow = 1)
  diag(weights_mat) <- NA

  fnmr <- map2(.x = fnmr_mat, .y = m_mat, .f = ~ sum(diag(.x) * diag(.y) * weights) / (sum(diag(.y) * weights))) %>%
    unlist()
  fmr <- map2(.x = fmr_mat, .y = m_mat, .f = ~ sum(.x * weights_mat * .y, na.rm = TRUE) / sum(weights_mat * .y, na.rm = TRUE)) %>%
    unlist()

  return(tibble(fnmr = fnmr, fmr = fmr))
}


get_bootstrap_vertex <- function(fnmr_mat, fmr_mat, m_mat) {
  # computes vertex bootstrap estimates of FNMR and FMR
  # fnmr_mat is a diagonal matrix with FNMR values per identity
  # fmr_mat is a matrix with FMR values per identity pair
  # m_mat is a matrix with the number of samples per identity pair
  if (!is.list(fmr_mat)) fmr_mat <- list(fmr_mat)
  if (!is.list(fnmr_mat)) fnmr_mat <- list(fnmr_mat)

  n_ids <- nrow(fnmr_mat[[1]])
  index <- sample(1:n_ids, size = n_ids, replace = TRUE)

  fnmr <- map2(.x = fnmr_mat, .y = m_mat, .f = ~ sum(diag(.x)[index] * diag(.y)[index]) / sum(diag(.y)[index])) %>%
    unlist()

  fmr <- c()
  for (i in 1:length(fmr_mat)) {
    fmr_mat_i <- fmr_mat[[i]][index, index]
    m_mat_i <- m_mat[[i]][index, index]

    m_mat_i[is.na(fmr_mat_i)] <- mean(m_mat_i, na.rm = TRUE)
    diag(m_mat_i) <- NA
    fmr_mat_i[is.na(fmr_mat_i)] <- mean(fmr_mat_i, na.rm = TRUE)
    diag(fmr_mat_i) <- NA

    fmr <- c(fmr, sum(fmr_mat_i * m_mat_i, na.rm = TRUE) / sum(m_mat_i, na.rm = TRUE))
  }

  return(tibble(fnmr = fnmr, fmr = fmr))
}



get_2levelboot <- function(genuine_scores, impostor_scores, m, thresholds) {
  # computes two-level bootstrap estimates of FNMR and FMR
  # genuine scores is a vector containing genuine scores
  # impostor scores is a vector containing impostor scores
  # m is a vector containing the number of samples per identity
  # n_ids is the number of identities
  # thresholds are the thresholds used for turning scores into binary predictions
  n_ids <- length(m)
  ids_sampled <- sample(1:n_ids, size = n_ids, replace = TRUE)
  images_sampled_by_id_fnmr <- ids_sampled %>% map(~ sample(1:(m[.x] * (m[.x] - 1)), size = m[.x] * (m[.x] - 1), replace = TRUE))
  images_sampled_by_id_fmr <- ids_sampled %>% map(~ sample(1:(m[.x] * (sum(m) - m[.x])), size = m[.x] * (sum(m) - m[.x]), replace = TRUE))

  # FNMR
  genuine_scores <- map2(
    .x = ids_sampled,
    .y = images_sampled_by_id_fnmr, .f = ~ genuine_scores[[.x]][.y]
  ) %>%
    unlist()

  impostor_scores <- map2(
    .x = ids_sampled,
    .y = images_sampled_by_id_fmr, .f = ~ impostor_scores[[.x]][.y]
  ) %>%
    unlist()

  return(thresholds %>% map(~ tibble(fnmr = mean(genuine_scores > .x), fmr = mean(impostor_scores <= .x)) %>% mutate(threshold = .x)))
}


get_wilson <- function(G, alpha, sample_size, var_sa) {
  # computes the Wilson score interval for the error rate
  # G is an adjacency matrix of the size n_ids x n_ids with FNMR (diagonal) or FMR (off-diagonal)
  # alpha is the significance level
  # var_sa is the variance of the error rate
  # sample size is the number of samples used to estimate the error rate

  G_vec <- as.vector(na.omit(as.vector(G)))
  n_eff <- mean(G_vec) * (1 - mean(G_vec)) / var_sa
  if (is.na(n_eff) | n_eff <= 0) {
    n_eff <- sample_size
  }

  mu <- mean(G_vec, na.rm = TRUE)

  x <- sum(G_vec, na.rm = TRUE)
  z2 <- qnorm(1 - alpha / 2)^2
  bounds <- (mu + z2 / 2 / n_eff + c(-1, 1) * qnorm(1 - alpha / 2) *
    sqrt((mu * (1 - mu) + z2 / 4 / n_eff) / n_eff)) / (1 + z2 / n_eff)
  if (x == 1) {
    bounds[1] <- -log(1 - alpha) / sample_size
  }
  if (x == n_eff - 1) {
    bounds[2] <- 1 + log(1 - alpha) / sample_size
  }
  return(pmax(bounds, 0))
}


## compute variance estimates ----

get_sample_analogue_var <- function(genuine_mat_acc, impostor_mat_acc, fnmr_mat, fmr_mat, m, m_mat = NULL) {
  # computes sample analogue variance estimates of FNMR and FMR
  # genuine_mat_acc is a matrix with genuine accuracy values per identity pair
  # impostor_mat_acc is a matrix with impostor accuracy values per identity pair
  # fnmr_mat is a diagonal matrix with FNMR values per identity
  # fmr_mat is a matrix with FMR values per identity pair
  # m is a vector with the number of samples per identity
  # m_mat is a matrix with the number of samples per identity pair

  sample_size <- sum(!is.na(impostor_mat_acc), na.rm = TRUE) / 2
  errors <- sum(impostor_mat_acc, na.rm = TRUE) / 2
  p <- errors / sample_size
  naive_var_fmr <- p * (1 - p) / sample_size
  n_ids <- length(m)

  fmr_mat <- fmr_mat - mean(impostor_mat_acc, na.rm = TRUE)
  fnmr_mat <- fnmr_mat - mean(genuine_mat_acc, na.rm = TRUE)
  genuine_mat_acc <- genuine_mat_acc - mean(genuine_mat_acc, na.rm = TRUE)
  impostor_mat_acc <- impostor_mat_acc - mean(impostor_mat_acc, na.rm = TRUE)

  # FNMR variance
  var_fnmr <- sum((diag(fnmr_mat) * (m * (m - 1) / 2))^2) / sum(m * (m - 1) / 2)^2

  # FMR variance
  var_fmr <- get_sample_analogue_var_fmr(impostor_mat_acc = impostor_mat_acc, fmr_mat = fmr_mat, m_mat = m_mat, m = m)
  var_fmr <- max(naive_var_fmr, var_fmr)


  return(list(var_fnmr = var_fnmr, var_fmr = var_fmr))
}


get_sample_analogue_var_fmr <- function(impostor_mat_acc, fmr_mat, m_mat, m) {
  # computes FMR variance estimate
  fmr_mat <- fmr_mat - mean(impostor_mat_acc, na.rm = TRUE)
  impostor_mat_acc <- impostor_mat_acc - mean(impostor_mat_acc, na.rm = TRUE)
  diag(m_mat) <- NA

  var_est <- c()
  cov_est <- c()

  G <- fmr_mat * m_mat
  diag(G) <- NA
  n_ids <- length(m)
  for (id in 1:n_ids) {
    for (id2 in setdiff(1:n_ids, id)) {
      var_est <- c(var_est, G[id, id2]^2)
      cov_est <- c(cov_est, sum(G[id, id2] * G[id, -c(id, id2)], na.rm = TRUE))
    }
  }
  var_fmr <- (2 * sum(var_est, na.rm = TRUE) + 4 * max(sum(cov_est, na.rm = TRUE), 0)) / sum(m_mat, na.rm = TRUE)^2

  return(var_fmr)
}


get_jackknife_var <- function(fnmr_mat, fmr_mat) {
  # computes jackknife variance estimate for FNMR and FMR
  n_ids <- nrow(fnmr_mat)

  # FNMR
  fnmr_except_id <- 1:n_ids %>%
    map(~ mean(diag(fnmr_mat)[-.x], na.rm = TRUE)) %>%
    unlist()
  var_fnmr_ind <- var(diag(fnmr_mat)) / n_ids
  fnmr <- mean(diag(fnmr_mat))
  var_fnmr <- (n_ids - 1) / n_ids * sum((fnmr_except_id - fnmr)^2)

  # FMR
  fmr_except_id <- 1:n_ids %>%
    map(~ mean(fmr_mat[-.x, -.x], na.rm = TRUE)) %>%
    unlist()
  fmr_vec <- as.vector(na.omit(as.vector(fmr_mat)))
  fmr <- mean(fmr_vec)
  var_fmr <- (n_ids - 2)^2 / n_ids^2 * sum((fmr_except_id - fmr)^2) - 2 * mean((fmr_mat - fmr)^2, na.rm = TRUE) / (n_ids * (n_ids - 1))

  return(tibble(var_fnmr = var_fnmr, var_fmr = var_fmr))
}


# generate embeddings -----

gen_data_emb <- function(n_ids, m, d,
                         distribution_id = "gaussian",
                         sigma2within = 1,
                         sigma2between = 1,
                         m_fixed = FALSE) {
  # generates embeddings and computes the adjacency matrix for downstream analysis
  # n_ids number of identities in the data
  # m number of samples per identity
  # d dimensionality of the embeddings
  # distribution_id distribution of the embeddings centers
  # sigma2within variance of the distribution of the embeddings within identities
  # sigma2between variance of the distribution of the embeddings centers
  # m_fixed whether m is fixed or not

  if(length(m_fixed) > 1){
     m = m_fixed
  } else if(m_fixed){
     m <- rep(m, n_ids)
  } else{
     m <- 1:n_ids %>% map_dbl(~ sample((m - 2):(m + 2), 1))
  }
  n <- sum(m)


  if (distribution_id == "gaussian") {
    mu_id <- matrix(rnorm(n_ids * d, mean = 0, sd = sqrt(sigma2between)), ncol = d)
    if (d == 1) mu_id <- matrix(mu_id, ncol = 1)
    stopifnot(m %% 1 == 0)
  } else if (distribution_id == "uniform") {
    mu_id <- matrix(runif(n_ids * d, -1, 1), nrow = n_ids, ncol = d)
  } else if (distribution_id == "lognormal") {
    mu_id <- matrix(rlnorm(n_ids * d, meanlog = 2, sdlog = sqrt(sigma2between)), ncol = d)
    if (d == 1) mu_id <- matrix(mu_id, ncol = 1)
  } else if (distribution_id == "exponential") {
    mu_id <- matrix(rexp(n_ids * d, rate = 1), nrow = n_ids, ncol = d)
  }

  gt <- 1:n_ids %>%
    map(~ rep(.x, each = m[.x])) %>%
    unlist()
  noise <- matrix(rnorm(sum(m) * d, mean = 0, sd = sqrt(sigma2within)), ncol = d)
  if (d == 1) noise <- matrix(noise, ncol = 1)
  emb <- mu_id[gt, ] + noise

  if (dim(emb)[2] > 1) {
    emb <- emb / apply(emb, 1, function(x) sqrt(sum(x^2)))
  }

  emb_mat <- as.matrix(dist(x = emb, p = 2))
  diag(emb_mat) <- NA

  genuine_mat_scores <- emb_mat
  impostor_mat_scores <- emb_mat
  for (id in 1:n_ids) {
    if (id == 1) pos_id <- 1:m[1]
    if (id > 1) pos_id <- (cumsum(m)[id - 1] + 1):cumsum(m)[id]
    genuine_mat_scores[pos_id, -c(pos_id)] <- NA
    impostor_mat_scores[pos_id, pos_id] <- NA
  }

  return(list(
    emb = emb,
    gt = gt,
    emb_mat = emb_mat,
    m = m,
    genuine_mat_scores = genuine_mat_scores,
    impostor_mat_scores = impostor_mat_scores
  ))
}


# other utils ----

## backbone of experiments ----

get_ci_for_fixed_threshold <- function(G, alpha, thresholds, methods = "all") {
  # computes confidence intervals for the FMR and FMR for a fixed threshold(s)
  # G is an adjancency matrix of dimension equal to the number of samples
  # alpha is the significance level
  # thresholds is a vector of thresholds
  # methods is a vector of methods to be used for computing the confidence intervals
  n_ids <- length(G$m)
  n <- sum(G$m) / 2

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

    sample_analogue_var <- get_sample_analogue_var(genuine_mat_acc = genuine_mat_acc, impostor_mat_acc = impostor_mat_acc, fnmr_mat = fnmr_mat, fmr_mat = fmr_mat, m = G$m, m_mat = m_mat)
    fnmr_wilson_bounds <- alpha %>%
      map_dfr(~ get_wilson(
        G = genuine_mat_acc,
        alpha = .x,
        sample_size = sum(!is.na(genuine_mat_acc)) / 2,
        var_sa = sample_analogue_var$var_fnmr
      ) %>% setNames(c("lb_fnmr", "ub_fnmr"))) %>%
      mutate(alpha = alpha)

    fmr_wilson_bounds <- alpha %>%
      map_dfr(~ get_wilson(
        G = impostor_mat_acc,
        alpha = .x,
        sample_size = sum(!is.na(impostor_mat_acc)) / 2, # n, # n_ids,
        var_sa = sample_analogue_var$var_fmr
      ) %>% setNames(c("lb_fmr", "ub_fmr"))) %>%
      mutate(alpha = alpha)

    out_wilson <- sample_analogue_var %>%
      bind_cols(fnmr_wilson_bounds) %>%
      inner_join(fmr_wilson_bounds, by = "alpha") %>%
      mutate(type = "wilson")

    ## wilson under data independence
    # FMR
    sample_size <- sum(!is.na(impostor_mat_acc), na.rm = TRUE) / 2
    errors <- sum(impostor_mat_acc, na.rm = TRUE) / 2
    fmr_wilson_bounds_ind <- alpha %>%
      map_dfr(~ Hmisc::binconf(errors, sample_size,
        alpha = .x,
        method = c("wilson"), # exact -> Clopper Pearson
        include.x = FALSE, include.n = FALSE, return.df = FALSE
      ) %>% data.frame()) %>%
      mutate(alpha = alpha) %>%
      setNames(c("est", "lb_fmr", "ub_fmr", "alpha"))
    # FNMR
    sample_size <- sum(!is.na(genuine_mat_acc), na.rm = TRUE) / 2
    errors <- sum(genuine_mat_acc, na.rm = TRUE) / 2
    fnmr_wilson_bounds_ind <- alpha %>%
      map_dfr(~ Hmisc::binconf(errors, sample_size,
        alpha = .x,
        method = c("wilson"),
        include.x = FALSE, include.n = FALSE, return.df = FALSE
      ) %>% data.frame()) %>%
      mutate(alpha = alpha) %>%
      setNames(c("est", "lb_fnmr", "ub_fnmr", "alpha"))
    out_wilson_ind <- fnmr_wilson_bounds_ind %>%
      inner_join(fmr_wilson_bounds_ind, by = "alpha") %>%
      select(-contains("est")) %>%
      mutate(type = "wilson_ind")

    wilson_ci <- wilson_ci %>% bind_rows(
      out_wilson %>% bind_rows(out_wilson_ind) %>% mutate(threshold = thresholds[i])
    )
  }

  # bootstraps
  out_boot <- tibble()
  n_boot <- 1e3
  if ("subset" %in% methods | "all" %in% methods) {
    out_boot_subset <- 1:n_boot %>%
      map_dfr(~ get_subsetboot(fmr_mat = out %>% map(~ .x$fmr_mat), fnmr_mat = out %>% map(~ .x$fnmr_mat)) %>% mutate(threshold = thresholds))
    out_boot <- out_boot %>% bind_rows(out_boot_subset %>% mutate(type = "subset"))
  }

  if ("db" %in% methods | "all" %in% methods) {
    out_db <- 1:n_boot %>%
      map_dfr(~ get_bootstrap_db(fmr_mat = out %>% map(~ .x$fmr_mat), fnmr_mat = out %>% map(~ .x$fnmr_mat), m_mat = out %>% map(~ .x$m_mat)) %>% mutate(threshold = thresholds))
    out_boot <- out_boot %>% bind_rows(out_db %>% mutate(type = "db"))
  }

  if ("vertex" %in% methods | "all" %in% methods) {
    out_vertex <- 1:n_boot %>%
      map_dfr(~ get_bootstrap_vertex(fmr_mat = out %>% map(~ .x$fmr_mat), fnmr_mat = out %>% map(~ .x$fnmr_mat), m_mat = out %>% map(~ .x$m_mat)) %>% mutate(threshold = thresholds))
    out_boot <- out_boot %>% bind_rows(out_vertex %>% mutate(type = "vertex"))
  }

  if ("2level" %in% methods | "all" %in% methods) {
    m <- G$m
    # speed up the bootstrap and feed it directly the vectors for each id
    1:n_ids %>% map(~ G$genuine_mat_scores[(c(0, cumsum(m)[1:(n_ids - 1)])[.x] + 1):(cumsum(m)[.x]), ] %>% discard(., is.na)) -> genuine_scores_by_id
    1:n_ids %>% map(~ G$impostor_mat_scores[(c(0, cumsum(m)[1:(n_ids - 1)])[.x] + 1):(cumsum(m)[.x]), ] %>% discard(., is.na)) -> impostor_scores_by_id

    out_2levelboot <- 1:n_boot %>%
      map_dfr(~ get_2levelboot(
        genuine_scores = genuine_scores_by_id,
        impostor_scores = impostor_scores_by_id,
        m = m,
        thresholds = thresholds
      ))
    out_boot <- out_boot %>% bind_rows(out_2levelboot %>% mutate(type = "2level"))
  }

  out <- wilson_ci

  if (length(methods) > 0) {
    out_boot <- out_boot %>%
      group_by(type, threshold) %>%
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
    out <- out %>%
      bind_rows(out_boot)
  }


  out %>%
    inner_join(tibble(current_fmr = current_fmr, current_fnmr = current_fnmr, threshold = thresholds), by = "threshold") %>%
    return()
}



get_coverage <- function(n_ids, m, d, distribution_id, sigma2between, sigma2within, alpha, thresholds, m_fixed = TRUE, methods = "all") {
  # get confidence intervals for a given set of parameters for synthetic data
  G <- gen_data_emb(n_ids = n_ids, m = m, d = d, distribution_id = distribution_id, sigma2within = sigma2within, sigma2between = sigma2between, m_fixed = m_fixed)

  out <- get_ci_for_fixed_threshold(G, alpha, thresholds, methods)

  return(list(out = out, roc_curve = NA))
}


get_metrics_split <- function(df, n_ids, alpha, thresholds, methods = "all") {
  # compute confidence intervals for real-world data
  df <- df %>% arrange(id_num)
  m <- df %>%
    count(id_num) %>%
    pull(n)
  embeddings <- df %>%
    select(matches("feature_"))

  G <- get_adj_mat(embeddings, m)
  cis <- get_ci_for_fixed_threshold(G, alpha, thresholds, methods)

  cis
}


## obtain vectors of scores ----

get_genuine_and_impostor_mat <- function(G, threshold) {
  # turn scores into binary predictions and compute errors for each combination of images
  # G is an adjancency matrix of dimension equal to the number of samples^2
  # threshold to be used for turning scores into binary predictions
  genuine_mat_acc <- G$genuine_mat_scores
  genuine_mat_acc[genuine_mat_acc > threshold] <- 0
  genuine_mat_acc[genuine_mat_acc > 0] <- 1

  impostor_mat_acc <- G$impostor_mat_scores
  impostor_mat_acc[impostor_mat_acc < threshold] <- 0
  impostor_mat_acc[impostor_mat_acc > 0] <- 1
  list(genuine_mat_acc = 1 - genuine_mat_acc, impostor_mat_acc = 1 - impostor_mat_acc)
}


compute_matrix_at_id_level <- function(genuine_mat_acc, impostor_mat_acc, m) {
  # compute FMR and FNMR at the ID level
  # genuine_mat_acc and impostor_mat_acc are matrices of size equal to the number of samples^2
  # n_ids is the number of IDs
  # m is the number of samples per ID
  n_ids <- length(m)
  id_start_idx <- cumsum(c(0, m[-length(m)])) + 1
  id_end_idx <- cumsum(m)

  fnmr_mat <- matrix(NA, nrow = n_ids, ncol = n_ids)
  fmr_mat <- matrix(NA, nrow = n_ids, ncol = n_ids)
  m_mat <- matrix(0, nrow = n_ids, ncol = n_ids)
  for (id_row in 1:n_ids) {
    for (id_col in 1:n_ids) {
      idx_row_current <- id_start_idx[id_row]:id_end_idx[id_row]
      idx_col_current <- id_start_idx[id_col]:id_end_idx[id_col]

      if (id_row == id_col) {
        idx_current <- id_start_idx[id_row]:id_end_idx[id_row]
        fnmr_mat[id_row, id_col] <- mean(genuine_mat_acc[idx_current, idx_current], na.rm = TRUE)
        m_mat[id_row, id_col] <- sum(!is.na(genuine_mat_acc[idx_current, idx_current]))
      } else {
        fmr_mat[id_row, id_col] <- mean(impostor_mat_acc[idx_row_current, idx_col_current], na.rm = TRUE)
        m_mat[id_row, id_col] <- sum(!is.na(impostor_mat_acc[idx_row_current, idx_col_current]))
      }
    }
  }

  return(list(fmr_mat = fmr_mat, fnmr_mat = fnmr_mat, m_mat = m_mat))
}


get_adj_mat <- function(embeddings, m) {
  # compute the adjacency matrix (l2 distances) for a given set of embeddings
  n_ids <- length(m)
  emb_mat <- as.matrix(dist(x = embeddings, p = 2))
  diag(emb_mat) <- NA

  genuine_mat_scores <- emb_mat
  impostor_mat_scores <- emb_mat
  pos_ids <- c(0, cumsum(m))
  for (id in 1:n_ids) {
    pos_id <- (pos_ids[id] + 1):(pos_ids[id + 1])
    genuine_mat_scores[pos_id, -pos_id] <- NA
    impostor_mat_scores[pos_id, pos_id] <- NA
  }

  return(list(
    emb_mat = emb_mat,
    genuine_mat_scores = genuine_mat_scores,
    impostor_mat_scores = impostor_mat_scores,
    m = m,
    gt = 1:n_ids %>% map(~ rep(.x, m[.x])) %>% unlist()
  ))
}


get_genuine_embeddings_x_id <- function(embeddings) {
  # compute the genuine scores for a given set of embeddings
  scores <- as.matrix(dist(embeddings, p = 2))
  diag(scores) <- NA
  scores[lower.tri(scores)] <- NA
  as.vector(scores) %>%
    na.omit() %>%
    as.vector()
}



## estimate thresholds -----


get_thresholds_for_targets <- function(n_ids, m, d, distribution_id, sigma2between, sigma2within, target_fmr, target_fnmr) {
  # get thresholds for a given set of parameters
  G <- gen_data_emb(n_ids = n_ids, m = m, d = d, distribution_id = distribution_id, sigma2within = sigma2within, sigma2between = sigma2between)
  fnmr_out <- target_fnmr %>% map_dfr(~ estimate_threshold_target_fnmr(G, .x) %>%
    mutate(target = "fnmr") %>%
    mutate(target_value = .x))
  fmr_out <- target_fmr %>% map_dfr(~ estimate_threshold_target_fmr(G, .x) %>%
    mutate(target = "fmr") %>%
    mutate(target_value = .x))
  fnmr_out %>% bind_rows(fmr_out)
}


estimate_threshold_target_fmr <- function(G, target_fmr) {
  # estimate threshold for the scores needed to achieve a target FMR level
  impostor_scores <- G$impostor_mat_scores
  genuine_scores <- G$genuine_mat_scores
  impostor_scores <- as.vector(na.omit(as.vector(impostor_scores))) %>%
    unique() %>%
    sort()

  fmr_levels <- seq(1, length(impostor_scores)) / length(impostor_scores)
  index <- which(fmr_levels <= target_fmr)
  if (sum(index) > 0) {
    index <- max(index)
    if (index == length(impostor_scores)) {
      th <- impostor_scores[index] + 1e-10
    } else {
      th <- (impostor_scores[index] + impostor_scores[index + 1]) / 2
    }
  } else {
    th <- min(impostor_scores) - 1e-10
  }

  return(tibble(threshold = th, fmr = mean(impostor_scores < th, na.rm = TRUE), fnmr = mean(genuine_scores >= th, na.rm = TRUE)))
}


estimate_threshold_target_fnmr <- function(G, target_fnmr) {
  # estimate threshold for the scores needed to achieve a target FNMR level
  impostor_scores <- G$impostor_mat_scores
  genuine_scores <- G$genuine_mat_scores
  genuine_scores <- as.vector(na.omit(as.vector(genuine_scores))) %>%
    sort()

  fnmr_levels <- seq(length(genuine_scores), 1) / length(genuine_scores)
  index <- which(fnmr_levels <= target_fnmr)
  if (sum(index)) {
    index <- min(index)
    th <- (genuine_scores[index] + genuine_scores[index + 1]) / 2
  } else {
    th <- max(genuine_scores) + 1e-10
  }

  return(tibble(threshold = th, fnmr = mean(genuine_scores >= th), fmr = mean(impostor_scores < th, na.rm = TRUE)))
}


estimate_threshold_target_fnmr_fromtor <- function(embeddings, target_fnmr, n_ids_fnmr, m) {
  # computes the threshold for a given target FNMR
  # this function allows to take into account more scores than get_thresholds_for_targets for FNMR computations
  pos_ids <- c(0, cumsum(m))
  genuine_scores <- 1:n_ids_fnmr %>%
    map(~ get_genuine_embeddings_x_id(embeddings[(pos_ids[.x] + 1):pos_ids[.x + 1], ])) %>%
    unlist() %>%
    sort()

  fnmr_levels <- seq(length(genuine_scores), 1) / length(genuine_scores)
  index <- which(fnmr_levels <= target_fnmr)
  if (sum(index)) {
    index <- min(index)
    th <- (genuine_scores[index] + genuine_scores[index + 1]) / 2
  } else {
    th <- max(genuine_scores) + 1e-10
  }

  return(tibble(threshold = th, fnmr = mean(genuine_scores >= th), fmr = NA))
}

get_thresholds_for_targets_separate <- function(df, target_fmr, target_fnmr, n_ids_fnmr, n_ids_fmr) {
  # compute thresholds for FNMR and FMR for a given set of embeddings
  # while get_thresholds_for_targets computes the thresholds for both FNMR and FMR on the same adjacency matrix, this computes them separately

  # FNMR threshold estimation
  ids_sampled <- sample(unique(df$id_num), size = n_ids_fnmr, replace = FALSE)
  dfs <- df %>%
    filter(id_num %in% ids_sampled) %>%
    arrange(id_num)
  m <- dfs %>%
    count(id_num) %>%
    pull(n)
  embeddings <- dfs %>%
    select(matches("feature_"))

  fnmr_out <- target_fnmr %>%
    map_dfr(~ estimate_threshold_target_fnmr_fromtor(
      embeddings = embeddings,
      target_fnmr = .x,
      m = m
    ) %>%
      mutate(target = "fnmr") %>%
      mutate(target_value = .x))

  # FMR threshold estimation
  ids_sampled <- sample(unique(df$id_num), size = n_ids_fmr, replace = FALSE)
  dfs <- df %>%
    filter(id_num %in% ids_sampled) %>%
    arrange(id_num)
  m <- dfs %>%
    count(id_num) %>%
    pull(n)
  embeddings <- dfs %>%
    select(matches("feature_"))

  G <- get_adj_mat(embeddings = embeddings, m = m)
  fmr_out <- target_fmr %>% map_dfr(~ estimate_threshold_target_fmr(G, .x) %>%
    mutate(target = "fmr") %>%
    mutate(target_value = .x))

  fnmr_out %>% bind_rows(fmr_out)
}
