import pytest
import torch
import torch.nn.functional as F

from backend.app.ai.segmentation_loss import CrossEntropyDiceLoss
from tests.segmentation_training_fixtures import deterministic_cpu


def test_cross_entropy_matches_torch_when_dice_disabled():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    assert torch.allclose(CrossEntropyDiceLoss(dice_weight=0)(logits, target), F.cross_entropy(logits, target))


@pytest.mark.parametrize("focal", [None, 0.0, 2.0])
def test_loss_finite_gradients_for_good_and_defect_masks(focal):
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[1, 2:5, 3:7] = 1
    loss = CrossEntropyDiceLoss(focal_gamma=focal)(logits, target)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_focal_zero_equals_ce_and_dice():
    logits, target = torch.randn(2, 2, 8, 8), torch.randint(0, 2, (2, 8, 8))
    assert torch.equal(CrossEntropyDiceLoss()(logits, target), CrossEntropyDiceLoss(focal_gamma=0)(logits, target))


def test_ignore_pixels_do_not_affect_loss_or_gradient():
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :2] = 255
    criterion = CrossEntropyDiceLoss(ignore_index=255)
    initial = criterion(logits, target)
    changed = logits.detach().clone()
    changed[:, :, :2] = 100
    assert torch.equal(initial.detach(), criterion(changed, target))
    initial.backward()
    assert torch.count_nonzero(logits.grad[:, :, :2]) == 0


def test_correct_predictions_have_lower_loss_than_incorrect_predictions():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    correct = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 20 - 10
    criterion = CrossEntropyDiceLoss()
    assert criterion(correct, target) < 1e-4
    assert criterion(correct.flip(1), target) > 10


@pytest.mark.parametrize("failure", ["invalid_id", "all_ignore", "float_target", "wrong_shape", "nan_logits"])
def test_invalid_loss_inputs_fail_closed(failure):
    logits, target = torch.randn(1, 2, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long)
    if failure == "invalid_id":
        target[0, 0, 0] = 2
    elif failure == "all_ignore":
        target.fill_(255)
    elif failure == "float_target":
        target = target.float()
    elif failure == "wrong_shape":
        target = target[:, :4]
    else:
        logits.fill_(float("nan"))
    with pytest.raises(ValueError):
        CrossEntropyDiceLoss(ignore_index=255)(logits, target)


@pytest.mark.parametrize("config", [{"ce_weight": 0, "dice_weight": 0}, {"ce_weight": -1}, {"focal_gamma": -1}, {"dice_weight": float("nan")}])
def test_invalid_loss_configuration_rejected(config):
    with pytest.raises(ValueError):
        CrossEntropyDiceLoss(**config)
