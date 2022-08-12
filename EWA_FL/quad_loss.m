function loss = quad_loss(hat_y, y)

loss = (hat_y - y).^2;