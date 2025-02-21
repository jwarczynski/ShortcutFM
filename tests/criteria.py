import pytest
import torch
from unittest.mock import MagicMock, call

from torch import tensor

from shortcutfm.criteria import (
    FlowMatchingCriterion,
    SelfConditioningFlowMatchingCriterionDecorator,
    ConsistencyCrterion, VelocityConsistencyCrterion
)


class TestFlowMatchingCriterion:
    def setup_method(self):
        """Set up the test environment before each test."""
        self.model = MagicMock()
        self.y1 = torch.tensor(
            [[[0.4400, 0.1558, 0.6868],
              [0.5822, 0.8113, 0.1854],
              [0.3590, 0.8383, 0.6319],
              [0.2017, 0.2134, 0.2720]],

             [[0.5473, 0.0877, 0.9816],
              [0.5535, 0.7768, 0.8155],
              [0.0295, 0.9140, 0.1516],
              [0.5796, 0.5796, 0.7658]]]
        )
        self.y2 = torch.tensor(
            [[[0.6400, 0.1558, 0.6868],
              [0.7822, 0.8113, 0.1854],
              [0.4590, 0.8383, 0.6319],
              [0.2017, 0.2134, 0.2720]],

             [[0.2473, 0.0877, 0.9816],
              [0.1535, 0.7768, 0.8155],
              [0.4295, 0.9140, 0.1516],
              [0.0796, 0.5796, 0.7658]]]
        )
        self.y3 = torch.tensor(
            [[[0.2400, 0.1558, 0.6868],
              [0.1822, 0.8113, 0.1854],
              [0.0590, 0.8383, 0.6319],
              [0.4017, 0.2134, 0.2720]],

             [[0.3473, 0.0877, 0.9816],
              [0.5535, 0.7768, 0.8155],
              [0.4295, 0.9140, 0.1516],
              [0.0796, 0.5796, 0.7658]]]
        )

        self.model.side_effect = [self.y1, self.y2, self.y3]  # Mock model to return y1 and y2
        self.criterion = FlowMatchingCriterion(self.model, 2048)

        # Define explicit tensors of shape (2, 4, 3) for deterministic testing
        self.x_start = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]
        ])

        self.x_t = torch.tensor([
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9], [10.10, 11.11, 12.12]],
            [[13.13, 14.14, 15.15], [16.16, 17.17, 18.18], [19.19, 20.20, 21.21], [22.22, 23.23, 24.24]]
        ])

        self.noise = torch.tensor([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8], [1.9, 2.0, 2.1], [2.2, 2.3, 2.4]]
        ])

        self.t = torch.tensor([1024, 512])
        self.shortcut_size = torch.tensor([32, 16])
        self.input_ids_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])

    def test_criterion_initialization(self):
        """Test if the criterion initializes correctly."""
        assert self.criterion.model == self.model

    def test_flow_matching_criterion(self):
        """Test if the criterion correctly processes inputs and returns expected values."""
        x_start_ret, y_ret = self.criterion(
            self.x_start,
            self.x_t,
            self.noise,
            self.t,
            self.shortcut_size,
            self.input_ids_mask
        )

        assert torch.equal(x_start_ret, self.x_start), "Mismatch in x_start"
        assert torch.equal(y_ret, self.y1), "Mismatch in y"

    def test_creiterion_time_scaling(self):
        """Test if the criterion correctly scales time."""
        t_scaled = self.criterion.scale_t(self.t)
        assert torch.equal(t_scaled, torch.tensor([0.5, 0.25])), "Mismatch in scaled time"

    def test_criterion_x_t_interpolation(self):
        """Test if the criterion correctly interpolates x_t."""
        # x_t = x_start + (noise - x_start) * t
        expected = torch.tensor(
            [[[0.5500, 1.1000, 1.6500],
              [2.2000, 2.7500, 3.3000],
              [3.8500, 4.4000, 4.9500],
              [5.5000, 6.0500, 6.6000]],

             [[10.0750, 10.8500, 11.6250],
              [12.4000, 13.1750, 13.9500],
              [14.7250, 15.5000, 16.2750],
              [17.0500, 17.8250, 18.6000]]]
        )
        x_t = self.criterion.get_x_t(self.x_start, self.noise, self.t)
        print(x_t)
        assert torch.equal(x_t, expected), "Mismatch in interpolated x_t"

    def test_self_conditioning_flow_matching_decorator_with_sc(self):
        """Test if the self-conditioning flow matching decorator correctly processes inputs."""
        torch.random.manual_seed(0)  # ensure deterministic results resulting in self-conditioning
        decorator = SelfConditioningFlowMatchingCriterionDecorator(self.criterion, 0.5)
        x_start_ret, y_ret = decorator(
            x_start=self.x_start,
            x_t=self.x_t,
            t=self.t,
            noise=self.noise,
            shortuct_size=self.shortcut_size,
            input_ids_mask=self.input_ids_mask
        )

        assert len(self.criterion.model.call_args_list) == 2, "No self_conditioning cacll"
        assert torch.equal(x_start_ret, self.x_start), "Mismatch in x_start"
        assert torch.equal(y_ret, self.y2), "Mismatch in y"

    def test_self_conditioning_flow_matching_decorator_without_sc(self):
        """Test if the self-conditioning flow matching decorator correctly processes inputs."""
        torch.random.manual_seed(44)  # ensure deterministic results resulting in no self-conditioning
        decorator = SelfConditioningFlowMatchingCriterionDecorator(self.criterion, 0.5)
        x_start_ret, y_ret = decorator(
            x_start=self.x_start,
            x_t=self.x_t,
            t=self.t,
            noise=self.noise,
            shortuct_size=self.shortcut_size,
            input_ids_mask=self.input_ids_mask
        )

        # Verify how the model was called
        args, kwargs = self.criterion.model.call_args
        assert torch.equal(args[0],
                           tensor([[[1.1000, 2.2000, 3.3000, 1.0000, 2.0000, 3.0000],
                                    [4.4000, 5.5000, 6.6000, 4.0000, 5.0000, 6.0000],
                                    [7.7000, 8.8000, 9.9000, 0.0000, 0.0000, 0.0000],
                                    [10.1000, 11.1100, 12.1200, 0.0000, 0.0000, 0.0000]],

                                   [[13.1300, 14.1400, 15.1500, 13.0000, 14.0000, 15.0000],
                                    [16.1600, 17.1700, 18.1800, 16.0000, 17.0000, 18.0000],
                                    [19.1900, 20.2000, 21.2100, 19.0000, 20.0000, 21.0000],
                                    [22.2200, 23.2300, 24.2400, 0.0000, 0.0000, 0.0000]]]),
                           ), "Mismatch in model call x_t argument"

        assert torch.equal(args[1], tensor([1024, 512])), "Mismatch in model call t argument"
        assert args[2] == tensor(0), "Mismatch in model call d argument"

        assert len(self.criterion.model.call_args_list) == 1
        assert torch.equal(x_start_ret, self.x_start), "Mismatch in x_start"
        assert torch.equal(y_ret, self.y1), "Mismatch in y"

    def test_consistency_criterion(self):
        """Test if the consistency criterion correctly processes inputs."""
        criterion = ConsistencyCrterion(self.model, 2048)
        target, y = criterion(
            x_start=self.x_start,
            x_t=self.x_t,
            noise=self.noise,
            t=self.t,
            shortcut_size=self.shortcut_size,
            input_ids_mask=self.input_ids_mask
        )

        first_call_args, _ = self.criterion.model.call_args_list[0]
        second_call_args, _ = self.criterion.model.call_args_list[1]
        last_call_args, _ = self.criterion.model.call_args_list[2]

        assert torch.equal(first_call_args[2], torch.tensor([32, 16])), "Model not queried with correct shortcut_size"
        assert torch.equal(second_call_args[2], torch.tensor([32, 16])), "Model not queried with correct shortcut_size"
        assert torch.equal(last_call_args[2], torch.tensor([64, 32])), "Model not queried with correct shortcut_size"

        assert torch.equal(target[0, 2:], self.y2[0, 2:]), "Mismatch in target"
        assert torch.equal(target[1, 3:], self.y2[1, 3:]), "Mismatch in target"
        assert torch.equal(y, self.y3), "Mismatch in y"

    def test_velocity_consistency_criterion(self):
        criterion = VelocityConsistencyCrterion(self.model, 2048)
        target, y = criterion(
            x_start=self.x_start,
            x_t=self.x_t,
            noise=self.noise,
            t=self.t,
            shortcut_size=self.shortcut_size,
            input_ids_mask=self.input_ids_mask
        )

        first_call_args, _ = criterion.model.call_args_list[0]
        second_call_args, _ = criterion.model.call_args_list[1]
        last_call_args, _ = criterion.model.call_args_list[2]

        assert torch.equal(first_call_args[2], torch.tensor([32, 16])), "Model not queried with correct shortcut_size"
        assert torch.equal(second_call_args[2], torch.tensor([32, 16])), "Model not queried with correct shortcut_size"
        assert torch.equal(last_call_args[2], torch.tensor([64, 32])), "Model not queried with correct shortcut_size"

        assert torch.equal(y, self.y3), "Mismatch in y"


if __name__ == "__main__":
    pytest.main()
