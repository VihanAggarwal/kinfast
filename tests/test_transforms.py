# tests/test_transforms.py
import torch, math
from kinfast import transforms as T

def test_rpy_identity():
    R = T.rpy_to_matrix(torch.zeros(3))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)

def test_rpy_yaw_90():
    R = T.rpy_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    # yaw 90deg maps +x -> +y
    x = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(R @ x, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

def test_axis_angle_z_90():
    R = T.axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    x = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(R @ x, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

def test_make_and_invert_transform():
    R = T.axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.7))
    t = torch.tensor([1.0, 2.0, 3.0])
    M = T.make_transform(R, t)
    I = M @ T.invert_transform(M)
    assert torch.allclose(I, torch.eye(4), atol=1e-6)

def test_so3_log_roundtrip():
    axis = torch.tensor([0.2, -0.5, 0.8]); axis = axis / axis.norm()
    angle = torch.tensor(1.1)
    R = T.axis_angle_to_matrix(axis, angle)
    w = T.so3_log(R)
    assert torch.allclose(w, axis * angle, atol=1e-5)

def test_pose_error_zero_when_equal():
    M = T.make_transform(T.rpy_to_matrix(torch.tensor([0.1, 0.2, 0.3])),
                         torch.tensor([1.0, 0.0, -1.0]))
    e = T.pose_error(M, M)
    assert torch.allclose(e, torch.zeros(6), atol=1e-6)

def test_batched_shapes():
    rpy = torch.zeros(5, 3)
    R = T.rpy_to_matrix(rpy)
    assert R.shape == (5, 3, 3)
