import torch

from rl4co.models.zoo.lehd._env import LEHDVRPEnv


def _original_lehd_length(problems, order_node, order_flag):
    """Reference formula from NCO_code/.../LEHD/CVRP/VRPEnv.py."""
    problem_size = problems.shape[1] - 1
    flags_as_nodes = order_flag.clone()
    flags_as_nodes[order_flag <= 0.5] = order_node[order_flag <= 0.5]
    flags_as_nodes[order_flag > 0.5] = 0

    gather_shape = (-1, problem_size, 2)
    order_loc = problems.gather(1, order_node.unsqueeze(2).expand(*gather_shape))
    roll_node = order_node.roll(shifts=1, dims=1)
    roll_loc = problems.gather(1, roll_node.unsqueeze(2).expand(*gather_shape))

    flag_loc = problems.gather(1, flags_as_nodes.unsqueeze(2).expand(*gather_shape))
    order_lengths = (order_loc - flag_loc).square()

    flags_as_nodes[:, 0] = 0
    flag_loc = problems.gather(1, flags_as_nodes.unsqueeze(2).expand(*gather_shape))
    roll_lengths = (roll_loc - flag_loc).square()
    return (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)


def test_cal_length_matches_original_lehd_for_multiple_routes():
    # depot=(0,0), with two routes: 0->1->2->0 and 0->3->4->0
    problems = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 2.0], [2.0, 2.0]]]
    )
    order_node = torch.tensor([[1, 2, 3, 4]])
    order_flag = torch.tensor([[1, 0, 1, 0]])

    env = LEHDVRPEnv(data_path="unused")
    actual = env._cal_length(problems, order_node, order_flag)
    expected = _original_lehd_length(problems, order_node, order_flag)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([6.0 + 3 * 2**0.5]))
