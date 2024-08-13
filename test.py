from neuron import MLP

def train(epochs, input_size, hidden_layers, output_size, xs, ys, lr):

    n = MLP(input_size, hidden_layers + [output_size])
    for k in range(epochs):

        # Forward pass
        yp = [n(x) for x in xs]
        loss = sum((j - i)**2 for i, j in zip(ys, yp))
        
        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        
        #loss 
        for p in n.parameters():
            p.data += -lr * p.grad  # -learning rate to minimize the loss
        
        print(k, "loss" ,loss.data)
    
    return n

def predict(n, x):
    return n(x)

if __name__ == '__main__':
    xs = [
        [1.0, 2.0, 3.9],
        [2.1, 3.3, 1.4],
        [1.9, 2.4, 2.4],
        [0.5, 1.0, 1.5]
    ]
    ys = [1.1, -1.0, 1.8, 1.5]
    model = train(epochs=10, input_size=3, hidden_layers=[10, 10], output_size=1, xs=xs, ys=ys, lr=0.01)

    new_x = [1.5, 2.5, 3.0]
    prediction = predict(model, new_x)
    print(f'Prediction for {new_x}: {prediction}')
