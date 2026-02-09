# simulate_given_joint_and_expectations.R

set.seed(123)

# ----- 1) Inputs: joint probs P(S=s, D=d) -----
# Order used below: (S1,D0), (S2,D0), (S3,D0), (S1,D1), (S2,D1), (S3,D1)
joint_probs <- c(
  P_S1_D0 = 0.36,
  P_S2_D0 = 0.12,
  P_S3_D0 = 0.12,
  P_S1_D1 = 0.08,
  P_S2_D1 = 0.12,
  P_S3_D1 = 0.20
)
stopifnot(abs(sum(joint_probs) - 1) < 1e-9)

# levels
S_levels <- c("1", "2", "3")
D_levels <- c("0", "1")

# build a little lookup table for (S,D) combinations in the same order as joint_probs
joint_table <- data.frame(
  S = c("1","2","3","1","2","3"),
  D = c("0","0","0","1","1","1"),
  prob = joint_probs,
  stringsAsFactors = FALSE
)

# ----- 2) E[Y | S, D] (rows = D values 0,1; cols = S = 1,2,3) -----
# Provided by you:
# E(Y|S=1,D=0)=2, E(Y|S=2,D=0)=6, E(Y|S=3,D=0)=10
# E(Y|S=1,D=1)=4, E(Y|S=2,D=1)=8, E(Y|S=3,D=1)=14

EY_matrix <- matrix(
  c(
    # row for D=0: columns S=1,2,3
    4, 6, 10,
    # row for D=1: columns S=1,2,3
    0, 10, 12
  ),
  nrow = 2,
  byrow = TRUE
)
rownames(EY_matrix) <- c("D0","D1")
colnames(EY_matrix) <- c("S1","S2","S3")

# ----- 3) Theoretical derived quantities -----
# Marginal P(S)
P_S <- tapply(joint_table$prob, joint_table$S, sum)[S_levels]

# Conditional P(D | S)
P_D_given_S <- matrix(NA, nrow = length(S_levels), ncol = length(D_levels),
                      dimnames = list(S_levels, D_levels))
for (s in S_levels) {
  probs_for_s <- joint_table$prob[joint_table$S == s]
  # ordering of probs_for_s is D=0 then D=1 by construction
  P_D_given_S[s, ] <- probs_for_s / sum(probs_for_s)
}

# Theoretical overall E[Y]
# E[Y] = sum_{s,d} P(s,d) * E[Y | s,d]
theoretical_EY <- sum(joint_table$prob * c(
  # this vector must match order of joint_table (S1D0, S2D0, S3D0, S1D1, S2D1, S3D1)
  EY_matrix[1,1], EY_matrix[1,2], EY_matrix[1,3],
  EY_matrix[2,1], EY_matrix[2,2], EY_matrix[2,3]
))

# Theoretical E[Y | S] (marginalizing D)
E_Y_given_S_theory <- numeric(length(S_levels))
names(E_Y_given_S_theory) <- S_levels
for (s in S_levels) {
  idxs <- which(joint_table$S == s)
  E_Y_given_S_theory[s] <- sum(joint_table$prob[idxs] * c(EY_matrix[1,which(S_levels==s)],
                                                          EY_matrix[2,which(S_levels==s)])) / sum(joint_table$prob[idxs])
}

# Print theoretical quantities
cat("Theoretical marginal P(S):\n"); print(P_S)
cat("\nTheoretical P(D | S):\n"); print(P_D_given_S)
cat("\nE[Y | D,S] matrix (rows D0,D1; cols S1,S2,S3):\n"); print(EY_matrix)
cat("\nTheoretical overall E[Y]: ", theoretical_EY, "\n")
cat("\nTheoretical E[Y | S]:\n"); print(E_Y_given_S_theory)

# ----- 4) Simulation: sample (S,D) from the joint and then Y ~ Normal(E[Y|S,D], sd = normal_sd) -----
generate_sample <- function(n = 5000, normal_sd = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  # sample joint category indices 1..6 with probabilities joint_table$prob
  idx <- sample(seq_len(nrow(joint_table)), size = n, replace = TRUE, prob = joint_table$prob)
  S_sim <- joint_table$S[idx]
  D_sim <- joint_table$D[idx]
  Y_sim <- numeric(n)
  for (i in seq_len(n)) {
    s <- S_sim[i]
    d <- D_sim[i]
    d_row <- ifelse(d == "0", 1, 2)
    s_col <- which(S_levels == s)
    mu <- EY_matrix[d_row, s_col]
    Y_sim[i] <- rnorm(1, mean = mu, sd = normal_sd)
  }
  data.frame(S = factor(S_sim, levels = S_levels),
             D = factor(D_sim, levels = D_levels),
             Y = Y_sim,
             stringsAsFactors = FALSE)
}

# Example run
n <- 50000
normal_sd <- 1.0
df <- generate_sample(n = n, normal_sd = normal_sd, seed = 2026)

# ----- 5) Empirical summaries -----
cat("\nEmpirical marginal P(S) (from simulation):\n"); print(prop.table(table(df$S)))
cat("\nEmpirical marginal P(D):\n"); print(prop.table(table(df$D)))
cat("\nEmpirical E[Y | D, S]:\n"); print(aggregate(Y ~ D + S, data = df, FUN = mean))
cat("\nEmpirical overall mean(Y): ", mean(df$Y), "\n")
summary(lm(Y~D+as.factor(S), data = df))