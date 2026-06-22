import copy
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils import debug


def _to_numpy_targets(targets):
    return targets.detach().cpu().view(-1).numpy().astype(int)


def _extract_scores(predictions, apply_sigmoid_if_binary=True):
    predictions = predictions.detach().cpu()

    if predictions.ndim == 2:
        if predictions.shape[1] == 2:
            # Two-class output. Use argmax-compatible positive-class score.
            probs = torch.softmax(predictions, dim=1)
            return probs[:, 1].numpy()
        if predictions.shape[1] == 1:
            scores = predictions.view(-1)
            if apply_sigmoid_if_binary:
                scores = torch.sigmoid(scores)
            return scores.numpy()

    if predictions.ndim == 1:
        scores = predictions
        if apply_sigmoid_if_binary:
            scores = torch.sigmoid(scores)
        return scores.numpy()

    raise ValueError(f"Unexpected prediction shape: {tuple(predictions.shape)}")


def _scores_to_predictions(scores, threshold=0.5):
    return (np.asarray(scores) >= threshold).astype(int)


def _collect_scores_targets_and_loss(model, loss_function, num_batches, data_iter, device):
    model.eval()
    all_scores, all_targets, losses = [], [], []

    with torch.no_grad():
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.float().to(device)

            predictions = model(graph, cuda=(device.type == "cuda"), device=device)
            batch_loss = loss_function(predictions, targets)
            losses.append(batch_loss.detach().cpu().item())

            scores = _extract_scores(predictions, apply_sigmoid_if_binary=True)
            all_scores.extend(scores.tolist())
            all_targets.extend(_to_numpy_targets(targets).tolist())

    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return np.array(all_scores), np.array(all_targets), mean_loss


def _compute_metrics_from_scores(scores, targets, threshold=0.5):
    preds = _scores_to_predictions(scores, threshold)

    acc = accuracy_score(targets, preds) * 100
    precision = precision_score(targets, preds, zero_division=0) * 100
    recall = recall_score(targets, preds, zero_division=0) * 100
    f1 = f1_score(targets, preds, zero_division=0) * 100

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": preds,
    }


def _find_best_threshold(scores, targets):
    best_threshold = 0.5
    best_f1 = -1.0
    best_acc = -1.0

    for threshold in np.arange(0.05, 0.96, 0.05):
        metrics = _compute_metrics_from_scores(scores, targets, threshold=threshold)
        curr_f1 = metrics["f1"]
        curr_acc = metrics["accuracy"]

        # Prefer F1 first, then accuracy as tie-breaker.
        if curr_f1 > best_f1 or (curr_f1 == best_f1 and curr_acc > best_acc):
            best_f1 = curr_f1
            best_acc = curr_acc
            best_threshold = float(threshold)

    return best_threshold, best_f1, best_acc


def evaluate_loss(model, loss_function, num_batches, data_iter, device, threshold=0.5):
    scores, targets, mean_loss = _collect_scores_targets_and_loss(
        model=model,
        loss_function=loss_function,
        num_batches=num_batches,
        data_iter=data_iter,
        device=device,
    )
    metrics = _compute_metrics_from_scores(scores, targets, threshold=threshold)
    return mean_loss, metrics["accuracy"]


def evaluate_metrics(model, loss_function, num_batches, data_iter, device, threshold=0.5):
    scores, targets, mean_loss = _collect_scores_targets_and_loss(
        model=model,
        loss_function=loss_function,
        num_batches=num_batches,
        data_iter=data_iter,
        device=device,
    )
    metrics = _compute_metrics_from_scores(scores, targets, threshold=threshold)
    return (
        mean_loss,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


def _default_checkpoint_path(save_path):
    if save_path is None:
        return "best_model.pt"
    if save_path.endswith(".pt") or save_path.endswith(".pth"):
        return save_path
    return os.path.join(save_path, "best_model.pt")


def train(
    model,
    dataset,
    max_steps,
    dev_every,
    loss_function,
    optimizer,
    save_path,
    log_every,
    max_patience,
    device,
):
    debug("Start training")

    train_losses = []
    best_model = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    patience_counter = 0

    checkpoint_path = _default_checkpoint_path(save_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    model.train()

    for step_count in range(max_steps):
        model.train()
        graph, targets = dataset.get_next_train_batch()
        targets = targets.float().to(device)

        optimizer.zero_grad()
        predictions = model(graph, cuda=(device.type == "cuda"), device=device)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().cpu().item())

        if log_every and (step_count + 1) % log_every == 0:
            debug(
                "Step {:d}/{:d}\tTrain Loss {:.4f}".format(
                    step_count + 1,
                    max_steps,
                    float(np.mean(train_losses[-log_every:])),
                )
            )

        if (step_count + 1) % dev_every == 0:
            val_scores, val_targets, val_loss = _collect_scores_targets_and_loss(
                model=model,
                loss_function=loss_function,
                num_batches=dataset.initialize_valid_batch(),
                data_iter=dataset.get_next_valid_batch,
                device=device,
            )

            curr_threshold, curr_val_f1, curr_val_acc = _find_best_threshold(
                val_scores, val_targets
            )

            debug(
                "Validation at step {:d}\tLoss {:.4f}\tAcc {:.2f}\tF1 {:.2f}\tThr {:.2f}".format(
                    step_count + 1,
                    val_loss,
                    curr_val_acc,
                    curr_val_f1,
                    curr_threshold,
                )
            )

            if curr_val_f1 > best_val_f1:
                best_val_f1 = curr_val_f1
                best_threshold = curr_threshold
                best_model = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_model,
                        "best_threshold": best_threshold,
                        "best_val_f1": best_val_f1,
                    },
                    checkpoint_path,
                )
                patience_counter = 0
                debug(
                    "New best model saved\tVal F1 {:.2f}\tThr {:.2f}".format(
                        best_val_f1,
                        best_threshold,
                    )
                )
            else:
                patience_counter += 1
                debug(
                    "No improvement\tPatience {:d}/{:d}".format(
                        patience_counter,
                        max_patience,
                    )
                )

            if patience_counter >= max_patience:
                debug("Early stopping triggered")
                break

    if best_model is None:
        debug("No validation improvement recorded. Using final model.")
        best_model = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "model_state_dict": best_model,
                "best_threshold": best_threshold,
                "best_val_f1": best_val_f1,
            },
            checkpoint_path,
        )

    model.load_state_dict(best_model)
    debug("Training complete")
    return model, best_threshold


def test(model, dataset, loss_function, device, threshold=0.5):
    model = model.to(device)
    model.eval()

    test_batches = dataset.initialize_test_batch()
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_metrics(
        model=model,
        loss_function=loss_function,
        num_batches=test_batches,
        data_iter=dataset.get_next_test_batch,
        device=device,
        threshold=threshold,
    )

    debug(
        "{}\tTest Accuracy: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}\tThreshold: {:.2f}".format(
            model.__class__.__module__ + "/" + model.__class__.__name__,
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            threshold,
        )
    )

    return test_loss, test_acc, test_precision, test_recall, test_f1


def train_and_test(
    model,
    dataset,
    max_steps,
    dev_every,
    loss_function,
    optimizer,
    save_path,
    log_every,
    max_patience,
    device,
):
    model, best_threshold = train(
        model=model,
        dataset=dataset,
        max_steps=max_steps,
        dev_every=dev_every,
        loss_function=loss_function,
        optimizer=optimizer,
        save_path=save_path,
        log_every=log_every,
        max_patience=max_patience,
        device=device,
    )

    return test(
        model=model,
        dataset=dataset,
        loss_function=loss_function,
        device=device,
        threshold=best_threshold,
    )


def load_best_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("best_threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5

    model = model.to(device)
    model.eval()
    return model, threshold
