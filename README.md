# Optimizers
Most commonly used deep learning optimizers from scratch in PyTorch along with visualization.

## 1. Vanilla SGD
```
for i in range(epochs):
    shuffled = np.random.shuffle(data)
    for batch in get.batch(shuffled, bs):
        grads = compute.grads(batch,weight, loss_func)
        params -= lr*grads
```

## 2. SGD with momentum
```
v_new = 0 #init update
rho = 0.9 #set rho
for i in range(epochs):
    shuffled = np.random.shuffle(data)
    for batch in get.batch(shuffled, bs):
        v_new = rho*v_prev + lr*compute.grads(batch, weight, loss_func)
        params -= v_new
        v_prev = v_new
```

## 3. Adadelta/RMSProp
```
grad_squared = 0
rho = 0.9 # param for exponential smoothing on grad squares
for i in range(epochs):
    shuffled = np.random.shuffle(data)
    for batch in get.batch(shuffled, bs):
        grads = compute.grads(batch, weight, loss_func)
        grads_sqaured = rho*(grad_squared) + (1 - rho)*(grad*grad)
        params -= lr*(grads / (np.sqrt(grads_squared) + noise))
```

## 4. Adam
```
m = 0
v = 0
beta1 = 0.999
beta2 = 1e-8
t = 0

for i in range(epochs):
    t += 1
    shuffled = np.random.shuffle(data)
    for batch in get.batch(shuffled, bs):
        grads = compute.grads(batch, weight - rho*v_prev, loss_func)
        m = beta1*m + (1 - beta1)*grads
        v = beta2*v + (1 - beta2)*grads*grads
        m_hat = m / (1 - beta1**t) # bias correction for first moment
        v_hat = v / (1 - beta1**t) # bias correction for second moment
        params -= lr*m_hat/(np.sqrt(v_hat) + noise)
```
