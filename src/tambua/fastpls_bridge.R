#!/usr/bin/env Rscript
## The model is fitted and predicted by tkcaccia/fastPLS 0.4 itself.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L || !(args[[1L]] %in% c("train", "predict"))) {
    stop("Usage: fastpls_bridge.R train|predict input.csv model.rds output.csv")
}
if (!requireNamespace("fastPLS", quietly = TRUE) ||
    as.character(utils::packageVersion("fastPLS")) != "0.4") {
    stop("Tambua requires GitHub tkcaccia/fastPLS version 0.4")
}
mode <- args[[1L]]
input <- utils::read.csv(args[[2L]], check.names = FALSE,
                         stringsAsFactors = FALSE)
if (nrow(input) == 0L) stop("The fastPLS input is empty")
if (mode == "train") {
    if (!("label" %in% names(input))) stop("Training labels are missing")
    features <- setdiff(names(input), "label")
    x <- as.matrix(input[, features, drop = FALSE])
    storage.mode(x) <- "double"
    y <- factor(input$label)
    ncomp <- min(nlevels(y) - 1L, nrow(x) - 1L, ncol(x))
    if (ncomp < 1L) stop("At least two classes and two samples are required")
    model <- fastPLS::pls(x, y, ncomp = ncomp,
                          scaling = "autoscaling", method = "simpls",
                          classifier = "lda", backend = "cpu", seed = 314159L,
                          fit = FALSE)
    saveRDS(list(model = model, features = features,
                 classes = levels(y), components = ncomp,
                 fastpls_version = "0.4"), args[[3L]])
    utils::write.csv(data.frame(components = ncomp), args[[4L]], row.names = FALSE)
} else {
    bundle <- readRDS(args[[3L]])
    if (!identical(bundle$fastpls_version, "0.4")) {
        stop("Incompatible fitted fastPLS version")
    }
    if (!identical(names(input), bundle$features)) {
        stop("The inference feature columns differ from training")
    }
    x <- as.matrix(input)
    storage.mode(x) <- "double"
    prediction <- predict(bundle$model, x, backend = "cpu")
    labels <- as.character(prediction$Ypred[[1L]])
    if (length(labels) != nrow(x) || anyNA(labels) ||
        any(!(labels %in% bundle$classes))) {
        stop("fastPLS returned invalid class labels")
    }
    utils::write.csv(data.frame(predicted = labels), args[[4L]], row.names = FALSE)
}
