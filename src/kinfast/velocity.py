"""Frame velocities and accelerations.

Position comes from forward kinematics and the first derivative comes from the
Jacobian, both of which the library already had. The piece that was missing is
the second one: the acceleration of a frame needs Jdot qd, the rate of change
of the Jacobian along the motion, and writing that out by hand for a general
tree is where the sign errors live.

It is not written out here. Jdot qd is the directional derivative of the
Jacobian along qd, so autograd computes it from the Jacobian the library
already tests against finite differences:

    twist(q, qd)              = J(q) qd
    acceleration(q, qd, qdd)  = J(q) qdd + (dJ/dq . qd) qd

Both are stacked as (linear, angular) in the world frame, matching the row
order of the Jacobian, and both are batched.
"""
import torch

from kinfast.jacobian import jacobian


def _check(q, qd, name="qd"):
    if qd.shape != q.shape:
        raise ValueError(f"{name} must have the same shape as q, got "
                         f"{tuple(qd.shape)} and {tuple(q.shape)}")
    return qd.to(device=q.device, dtype=q.dtype)


def twist(chain, q, qd, link_index):
    """Spatial velocity of a link, (B, 6) as (linear, angular) in world axes.

    This is J(q) qd. The linear part is the velocity of the link's origin, not
    of its centre of mass, which matters when a link's frame sits away from
    its mass.
    """
    qd = _check(q, qd)
    J = jacobian(chain, q, link_index)
    return (J @ qd.unsqueeze(-1)).squeeze(-1)


def jacobian_dot_qd(chain, q, qd, link_index):
    """The (dJ/dq . qd) qd term, (B, 6), without deriving it by hand.

    torch.autograd.functional.jvp pushes the direction qd through the Jacobian
    to get Jdot, and the result is contracted with qd again. Doing it this way
    means the term inherits whatever the Jacobian is, so it cannot drift away
    from the kinematics the rest of the library uses.
    """
    qd = _check(q, qd)
    q0 = q.detach()

    def J_of(qq):
        return jacobian(chain, qq, link_index)

    _J, Jdot = torch.autograd.functional.jvp(J_of, q0, qd)
    return (Jdot @ qd.unsqueeze(-1)).squeeze(-1)


def acceleration(chain, q, qd, qdd, link_index):
    """Spatial acceleration of a link, (B, 6) as (linear, angular) in world axes.

    J qdd + Jdot qd. With qdd zero this is the acceleration a frame has purely
    from the motion already underway, which is the centripetal term and is not
    zero even at constant joint velocity.
    """
    qd = _check(q, qd)
    qdd = _check(q, qdd, "qdd")
    J = jacobian(chain, q, link_index)
    return ((J @ qdd.unsqueeze(-1)).squeeze(-1)
            + jacobian_dot_qd(chain, q, qd, link_index))


def link_velocities(chain, q, qd):
    """Twists of every link at once, (B, n_links, 6).

    Cheaper than calling twist once per link when a caller wants the whole
    tree, because the forward pass behind the Jacobians is shared.
    """
    qd = _check(q, qd)
    return torch.stack(
        [twist(chain, q, qd, i) for i in range(chain.n_links)], dim=1)
