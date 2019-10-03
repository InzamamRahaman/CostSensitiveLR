import torch as th


def cost_sensitive_loss(model_output, class_labels,  b00, b01, b10, b11):
    eta = th.tensor((b00 - b01) / (b11 - b10))
    p1 = -class_labels * th.log(model_output)
    p2 = -(th.tensor(1.0) - class_labels) * th.log(th.tensor(1.0) - model_output) * eta
    J_b_i = p1 + p2
    J_b = th.sum(J_b_i)
    return J_b
